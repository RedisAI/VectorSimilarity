/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include <arm_sve.h>

// SVE.cpp and SVE2.cpp both compile this header, under different -march flags. The
// anonymous namespace keeps each tier's bodies to itself; without it they are weak
// symbols that both objects define and link order picks the -march. Only the Choose_*
// entry points stay external. Dependencies above must stay outside the namespace.
namespace {

inline void InnerProduct_Step(const float16_t *vec1, const float16_t *vec2, svfloat16_t &acc,
                              size_t &offset, const size_t chunk) {
    svbool_t all = svptrue_b16();

    // Load half-precision vectors.
    svfloat16_t v1 = svld1_f16(all, vec1 + offset);
    svfloat16_t v2 = svld1_f16(all, vec2 + offset);
    // Compute multiplications and add to the accumulator
    acc = svmla_f16_x(all, acc, v1, v2);

    // Move to next chunk
    offset += chunk;
}

template <bool partial_chunk, unsigned char additional_steps> // [t/f, 0..7]
float FP16_InnerProduct_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *vec1 = static_cast<const float16_t *>(pVect1v);
    const auto *vec2 = static_cast<const float16_t *>(pVect2v);
    const size_t chunk = svcnth(); // number of 16-bit elements in a register
    svbool_t all = svptrue_b16();
    svfloat16_t acc1 = svdup_f16(0.0f);
    svfloat16_t acc2 = svdup_f16(0.0f);
    svfloat16_t acc3 = svdup_f16(0.0f);
    svfloat16_t acc4 = svdup_f16(0.0f);
    svfloat16_t acc5 = svdup_f16(0.0f);
    svfloat16_t acc6 = svdup_f16(0.0f);
    svfloat16_t acc7 = svdup_f16(0.0f);
    svfloat16_t acc8 = svdup_f16(0.0f);
    size_t offset = 0;

    // Eight accumulators shorten the native-fp16 FMA dependency chains. This improves instruction
    // parallelism and limits rounding drift on implementations with a short SVE vector length.
    const size_t full_iterations = dimension / chunk / 8;
    for (size_t iter = 0; iter < full_iterations; iter++) {
        InnerProduct_Step(vec1, vec2, acc1, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc2, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc3, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc4, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc5, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc6, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc7, offset, chunk);
        InnerProduct_Step(vec1, vec2, acc8, offset, chunk);
    }

    // Perform between 0 and 7 additional steps, according to `additional_steps` value
    if constexpr (additional_steps >= 1)
        InnerProduct_Step(vec1, vec2, acc1, offset, chunk);
    if constexpr (additional_steps >= 2)
        InnerProduct_Step(vec1, vec2, acc2, offset, chunk);
    if constexpr (additional_steps >= 3)
        InnerProduct_Step(vec1, vec2, acc3, offset, chunk);
    if constexpr (additional_steps >= 4)
        InnerProduct_Step(vec1, vec2, acc4, offset, chunk);
    if constexpr (additional_steps >= 5)
        InnerProduct_Step(vec1, vec2, acc5, offset, chunk);
    if constexpr (additional_steps >= 6)
        InnerProduct_Step(vec1, vec2, acc6, offset, chunk);
    if constexpr (additional_steps >= 7)
        InnerProduct_Step(vec1, vec2, acc7, offset, chunk);

    // Handle the tail with the residual predicate
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b16_u64(offset, dimension);

        // Load half-precision vectors.
        svfloat16_t v1 = svld1_f16(pg, vec1 + offset);
        svfloat16_t v2 = svld1_f16(pg, vec2 + offset);
        // Compute multiplications and add to the accumulator.
        // use the existing value of `acc` for the inactive elements (by the `m` suffix)
        acc8 = svmla_f16_m(pg, acc8, v1, v2);
    }

    // Accumulate accumulators
    acc1 = svadd_f16_x(all, acc1, acc5);
    acc2 = svadd_f16_x(all, acc2, acc6);
    acc3 = svadd_f16_x(all, acc3, acc7);
    acc4 = svadd_f16_x(all, acc4, acc8);
    acc1 = svadd_f16_x(all, acc1, acc3);
    acc2 = svadd_f16_x(all, acc2, acc4);
    acc1 = svadd_f16_x(all, acc1, acc2);

    // Reduce the accumulated sum.
    float result = svaddv_f16(all, acc1);
    return 1.0f - result;
}
} // namespace
