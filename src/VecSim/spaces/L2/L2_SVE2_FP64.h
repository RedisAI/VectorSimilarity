/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static void L2SquareStep_SVE2(double *&pVect1, double *&pVect2, size_t &offset, svfloat64_t &sum) {
    // Load vectors
    svfloat64_t v1 = svld1_f64(svptrue_b64(), pVect1 + offset);
    svfloat64_t v2 = svld1_f64(svptrue_b64(), pVect2 + offset);

    // Calculate difference between vectors
    svfloat64_t diff = svsub_f64_x(svptrue_b64(), v1, v2);

    // Square the difference and accumulate: sum += diff * diff
    sum = svmla_f64_z(svptrue_b64(), sum, diff, diff);

    // Advance pointers by the vector length
    offset += svcntd();
}

template <bool partial_chunk, unsigned char additional_steps>
double FP64_L2SqrSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double*) pVect1v;
    double *pVect2 = (double*) pVect2v;
    size_t offset = 0;

    // Get the number of 64-bit elements per vector at runtime
    uint64_t vl = svcntd();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat64_t sum0 = svdup_f64(0.0f);
    svfloat64_t sum1 = svdup_f64(0.0f);
    svfloat64_t sum2 = svdup_f64(0.0f);
    svfloat64_t sum3 = svdup_f64(0.0f);

    // Process vectors in chunks, with unrolling for better pipelining
    auto chunk_size = 4 * vl;
    size_t number_of_chunks = dimension / chunk_size;
    for (size_t i = 0; i < number_of_chunks; ++i) {
        // Process 4 chunks with separate accumulators
        L2SquareStep_SVE2(pVect1, pVect2, offset, sum0);
        L2SquareStep_SVE2(pVect1, pVect2, offset, sum1);
        L2SquareStep_SVE2(pVect1, pVect2, offset, sum2);
        L2SquareStep_SVE2(pVect1, pVect2, offset, sum3);
    }

    if constexpr (additional_steps > 0) {
        for (unsigned char c = 0; c < additional_steps; ++c) {
            L2SquareStep_SVE2(pVect1, pVect2, offset, sum0);
        }
    }
    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b64(offset, dimension);

        // Load vectors with predication
        svfloat64_t v1 = svld1_f64(pg, pVect1 + offset);
        svfloat64_t v2 = svld1_f64(pg, pVect2 + offset);

        // Calculate difference with predication (corrected)
        svfloat64_t diff = svsub_f64_m(pg, v1, v2);

        // Square the difference and accumulate with predication
        sum0 = svmla_f64_m(pg, sum0, diff, diff);
    }

    // Combine the partial sums
    sum0 = svadd_f64_z(svptrue_b64(), sum0, sum1);
    sum2 = svadd_f64_z(svptrue_b64(), sum2, sum3);
    sum0 = svadd_f64_z(svptrue_b64(), sum0, sum2);

    // Horizontal sum
    double result = svaddv_f64(svptrue_b64(), sum0);
    return result;
}
