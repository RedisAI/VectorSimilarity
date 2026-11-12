/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include "VecSim/utils/alignment.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/sq8_a8.h"

namespace spaces {
namespace quantized {

// Complete the symmetric stored-stored inner product from the integer code dot and stored sums.
// This ISA-neutral helper is shared by scalar and SIMD kernels. Internal linkage keeps each
// translation unit's body under its own ISA flags instead of merging target-specific definitions.
// The expression's operation order is part of the numerical contract: do not reassociate it or
// compile with -ffast-math / -Ofast.
template <typename StorageType>
static inline float Stored_InnerProduct_FromCodeDot(const uint8_t *a, const uint8_t *b, size_t dim,
                                                    float code_dot) {
    // Metadata follows byte payloads and is not necessarily float-aligned.
    const auto *params1 = a + StorageType::data_bytes_count(dim);
    const float min_val1 = load_unaligned<float>(params1 + StorageType::MIN_VAL * sizeof(float));
    const float delta1 = load_unaligned<float>(params1 + StorageType::DELTA * sizeof(float));
    const float sum1 = load_unaligned<float>(params1 + StorageType::SUM * sizeof(float));
    const auto *params2 = b + StorageType::data_bytes_count(dim);
    const float min_val2 = load_unaligned<float>(params2 + StorageType::MIN_VAL * sizeof(float));
    const float delta2 = load_unaligned<float>(params2 + StorageType::DELTA * sizeof(float));
    const float sum2 = load_unaligned<float>(params2 + StorageType::SUM * sizeof(float));

    return min_val1 * sum2 + min_val2 * sum1 - static_cast<float>(dim) * min_val1 * min_val2 +
           delta1 * delta2 * code_dot;
}

// Complete the SQ8 storage vs A8 query inner product distance from the integer code dot
// sum(a_i * p_i) (see vecsim_types::sq8_a8_query), in double:
//   distance = 1 - [(delta * QD) * dot + min * QM + SUM * QS + QK]
// Every tier calls this helper, so all tiers return the same float for the same dot. The fused
// steps are written as std::fma: the SIMD translation units are built with FMA-capable ISA flags
// and the compiler's default contraction, which would fuse a written a * b + c there but not in
// the scalar translation unit, and the tiers would then round differently.
static inline float SQ8_A8_InnerProduct_FromCodeDot(const uint8_t *stored, const uint8_t *query,
                                                    size_t dim, int64_t code_dot) {
    using sq8 = vecsim_types::sq8;
    using a8 = vecsim_types::sq8_a8_query;
    // Stored metadata follows the codes and is not necessarily float-aligned.
    const auto *params = stored + sq8::data_bytes_count(dim);
    const double min_val = load_unaligned<float>(params + sq8::MIN_VAL * sizeof(float));
    const double delta = load_unaligned<float>(params + sq8::DELTA * sizeof(float));
    const double sum = load_unaligned<float>(params + sq8::SUM * sizeof(float));
    const auto *consts = query + a8::consts_offset(dim);
    const double qd = load_unaligned<double>(consts + a8::QD * sizeof(double));
    const double qm = load_unaligned<double>(consts + a8::QM * sizeof(double));
    const double qs = load_unaligned<double>(consts + a8::QS * sizeof(double));
    const double qk = load_unaligned<double>(consts + a8::QK * sizeof(double));

    double ip = delta * qd * static_cast<double>(code_dot);
    ip = std::fma(min_val, qm, ip);
    ip = std::fma(sum, qs, ip);
    ip += qk;
    return static_cast<float>(1.0 - ip);
}

} // namespace quantized
} // namespace spaces
