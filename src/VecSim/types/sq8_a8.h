/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace vecsim_types {

// Query blob of an SQ8 IP / Cosine index whose query is quantized once to signed 8-bit codes
// ("A8"), so that each distance is one integer dot product against the stored uint8 codes.
//
// The query y is encoded on its own grid yh_i = c + s * p_i, with s = (max y - min y) / 255,
// c = min y + 128 * s and codes p_i in [-128, 127]. With e_i = y_i - yh_i, E = sum e_i,
// P = sum p_i and K = s * sum (p_i - P / dim) * e_i, the inner product with a stored vector
// (min m, delta d, SUM) is estimated as
//   IP = (d * QD) * dot + m * QM + SUM * QS + QK,   dot = sum a_i * p_i
// where E and K fold the query's own rounding error into per-query constants.
//
// Layout: [int8 codes (dim)] [zero padding up to a multiple of 8 bytes] [8-byte constant slots]
// The slots are QD, QM, QS, QK and Y_MEAN_IP as double, then P as int64.
struct sq8_a8_query {
    using value_type = int8_t;

    enum ConstIndex : size_t {
        QD = 0,        // s
        QM = 1,        // s * P
        QS = 2,        // c + E / dim
        QK = 3,        // K
        Y_MEAN_IP = 4, // <mean, y> for a mean-centered index, else 0; not read by the kernels
        P = 5,         // sum of the codes (int64), for kernels that offset the stored codes
        CONSTS_COUNT = 6
    };

    static constexpr size_t consts_offset(size_t dim) { return (dim + 7) & ~size_t(7); }

    static constexpr size_t blob_bytes_count(size_t dim) {
        return consts_offset(dim) + CONSTS_COUNT * sizeof(double);
    }

    // Allocation alignment of a query blob. The kernels need none (unaligned and masked loads,
    // constants read with memcpy); 64 keeps the codes on whole cache lines.
    static constexpr unsigned char blob_alignment = 64;
};

static_assert(sizeof(double) == sizeof(int64_t), "A8 query constants use 8-byte slots");

} // namespace vecsim_types
