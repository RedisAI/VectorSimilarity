/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <arm_sve.h>
#include <cmath>
#include <cstdint>
#include "VecSim/spaces/functions/SVE2.h"
#include "VecSim/types/sq8.h"
#include "VecSim/types/float16.h"
#include "VecSim/utils/alignment.h"

namespace {

inline void SQ8_FP16_InnerProductStep_SVE2_NATIVE(const uint8_t *codes, const float16_t *query,
                                                  svfloat16_t &acc, size_t &offset, size_t chunk) {
    const svbool_t all = svptrue_b16();
    const svuint16_t code_values = svld1ub_u16(all, codes + offset);
    const svfloat16_t values = svcvt_f16_u16_x(all, code_values);
    const svfloat16_t query_values = svld1_f16(all, query + offset);
    acc = svmla_f16_x(all, acc, values, query_values);
    offset += chunk;
}

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP16_InnerProductSIMD_SVE2_NATIVE(const void *storage, const void *query_blob,
                                            size_t dimension) {
    using sq8 = vecsim_types::sq8;
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    // Directed rounding can saturate half overflow, hiding it from the finite check.
    if (fpcr & (uint64_t{3} << 22))
        return spaces::Choose_SQ8_FP16_IP_implementation_SVE2(dimension)(storage, query_blob,
                                                                         dimension);

    const auto *codes = static_cast<const uint8_t *>(storage);
    const auto *query = static_cast<const float16_t *>(query_blob);
    const size_t chunk = svcnth();
    const svbool_t all = svptrue_b16();
    svfloat16_t acc1 = svdup_f16(0.0f);
    svfloat16_t acc2 = svdup_f16(0.0f);
    svfloat16_t acc3 = svdup_f16(0.0f);
    svfloat16_t acc4 = svdup_f16(0.0f);
    size_t offset = 0;

    const size_t full_iterations = dimension / chunk / 4;
    for (size_t iter = 0; iter < full_iterations; ++iter) {
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc1, offset, chunk);
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc2, offset, chunk);
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc3, offset, chunk);
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc4, offset, chunk);
    }
    if constexpr (additional_steps >= 1)
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc1, offset, chunk);
    if constexpr (additional_steps >= 2)
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc2, offset, chunk);
    if constexpr (additional_steps >= 3)
        SQ8_FP16_InnerProductStep_SVE2_NATIVE(codes, query, acc3, offset, chunk);

    if constexpr (partial_chunk) {
        const svbool_t pg = svwhilelt_b16_u64(offset, dimension);
        const svuint16_t code_values = svld1ub_u16(pg, codes + offset);
        const svfloat16_t values = svcvt_f16_u16_x(pg, code_values);
        const svfloat16_t query_values = svld1_f16(pg, query + offset);
        acc4 = svmla_f16_m(pg, acc4, values, query_values);
    }

    // Keep the grouping and half reduction identical to the dense FP16 SVE kernel.
    acc1 = svadd_f16_x(all, acc1, acc3);
    acc2 = svadd_f16_x(all, acc2, acc4);
    acc1 = svadd_f16_x(all, acc1, acc2);
    const float dot = svaddv_f16(all, acc1);
    if (!std::isfinite(dot))
        return spaces::Choose_SQ8_FP16_IP_implementation_SVE2(dimension)(storage, query_blob,
                                                                         dimension);

    const auto *storage_meta = codes + dimension;
    const float min_val = load_unaligned<float>(storage_meta + sq8::MIN_VAL * sizeof(float));
    const float delta = load_unaligned<float>(storage_meta + sq8::DELTA * sizeof(float));
    const auto *query_meta =
        static_cast<const uint8_t *>(query_blob) + dimension * sizeof(vecsim_types::float16);
    const float query_sum = load_unaligned<float>(query_meta + sq8::SUM_QUERY * sizeof(float));
    return 1.0f - (min_val * query_sum + delta * dot);
}

} // namespace
