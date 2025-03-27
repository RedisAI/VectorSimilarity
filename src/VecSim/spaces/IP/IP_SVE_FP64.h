/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

static void InnerProductStep(double *&pVect1, double *&pVect2, size_t &offset, svfloat64_t &sum) {
    // Load vectors
    svfloat64_t v1 = svld1_f64(svptrue_b64(), pVect1 + offset);
    svfloat64_t v2 = svld1_f64(svptrue_b64(), pVect2 + offset);

    // Multiply-accumulate
    sum = svmla_f64_z(svptrue_b64(), sum, v1, v2);

    // Advance pointers
    offset += svcntd();
}

template <bool partial_chunk, unsigned char additional_steps>
double FP64_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;
    size_t offset = 0;

    uint64_t vl = svcntd();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat64_t sum0 = svdup_f64(0.0f);
    svfloat64_t sum1 = svdup_f64(0.0f);
    svfloat64_t sum2 = svdup_f64(0.0f);
    svfloat64_t sum3 = svdup_f64(0.0f);

    auto chunk_size = 4 * vl;
    size_t number_of_chunks = dimension / chunk_size;
    for (size_t i = 0; i < number_of_chunks; i++) {
        InnerProductStep(pVect1, pVect2, offset, sum0);
        InnerProductStep(pVect1, pVect2, offset, sum1);
        InnerProductStep(pVect1, pVect2, offset, sum2);
        InnerProductStep(pVect1, pVect2, offset, sum3);
    }

    if constexpr (additional_steps > 0) {
        if constexpr (additional_steps >= 1) {
            InnerProductStep(pVect1, pVect2, offset, sum0);
        }
        if constexpr (additional_steps >= 2) {
            InnerProductStep(pVect1, pVect2, offset, sum1);
        }
        if constexpr (additional_steps >= 3) {
            InnerProductStep(pVect1, pVect2, offset, sum2);
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b64(offset, dimension);
        svfloat64_t v1 = svld1_f64(pg, pVect1 + offset);
        svfloat64_t v2 = svld1_f64(pg, pVect2 + offset);
        sum3 = svmla_f64_m(pg, sum3, v1, v2);
    }

    // Combine the partial sums
    sum0 = svadd_f64_z(svptrue_b64(), sum0, sum1);
    sum2 = svadd_f64_z(svptrue_b64(), sum2, sum3);

    // Perform vector addition in parallel
    svfloat64_t sum_all = svadd_f64_z(svptrue_b64(), sum0, sum2);
    // Single horizontal reduction at the end
    double result = svaddv_f64(svptrue_b64(), sum_all);
    return 1.0f - result;
}