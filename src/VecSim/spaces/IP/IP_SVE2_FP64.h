/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

static void InnerProductStep_SVE2(double *&pVect1, double *&pVect2, size_t &offset, svfloat64_t &sum) {
    // Load vectors
    svfloat64_t v1 = svld1_f64(svptrue_b64(), pVect1 + offset);
    svfloat64_t v2 = svld1_f64(svptrue_b64(), pVect2 + offset);

    // Multiply-accumulate
    sum = svmla_f64_z(svptrue_b64(), sum, v1, v2);

    // Advance pointers
    offset += svcntd();
}

template <bool partial_chunk, unsigned char additional_steps>
double FP64_InnerProductSIMD_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *pVect1 = (double*)pVect1v;
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
        InnerProductStep_SVE2(pVect1, pVect2, offset, sum0);
        InnerProductStep_SVE2(pVect1, pVect2, offset, sum1);
        InnerProductStep_SVE2(pVect1, pVect2, offset, sum2);
        InnerProductStep_SVE2(pVect1, pVect2, offset, sum3);
    }

    if constexpr (additional_steps > 0) {
        for (unsigned char c = 0; c < additional_steps; ++c) {
            InnerProductStep_SVE2(pVect1, pVect2, offset, sum0);
        }
    }

    if constexpr (partial_chunk) {
        svbool_t pg = svwhilelt_b64(offset, dimension);
        svfloat64_t v1 = svld1_f64(pg, pVect1 + offset);
        svfloat64_t v2 = svld1_f64(pg, pVect2 + offset);
        sum0 = svmla_f64_m(pg, sum0, v1, v2);
    }

    // Combine the partial sums
    sum0 = svadd_f64_z(svptrue_b64(), sum0, sum1);
    sum2 = svadd_f64_z(svptrue_b64(), sum2, sum3);
    sum0 = svadd_f64_z(svptrue_b64(), sum0, sum2);

    // Horizontal sum
    double result = svaddv_f64(svptrue_b64(), sum0);
    return 1.0f - result;
}
