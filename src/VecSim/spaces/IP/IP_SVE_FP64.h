/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

#include <arm_sve.h>

static void InnerProductStep(double *&pVect1, double *&pVect2, svfloat64_t &sum) {
    // Load vectors
    svfloat64_t v1 = svld1_f64(svptrue_b64(), pVect1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(), pVect2);

    // Multiply-accumulate
    sum = svmla_f64_z(svptrue_b64(), sum, v1, v2);

    // Advance pointers
    pVect1 += svcntd();
    pVect2 += svcntd();
}

template <unsigned char residual>
double FP64_InnerProductSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const double *pVect1 = (double*)pVect1v;
    const double *pVect2 = (double *)pVect2v;

    uint64_t vl = svcntd();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat64_t sum0 = svdup_f64(0.0f);
    svfloat64_t sum1 = svdup_f64(0.0f);
    svfloat64_t sum2 = svdup_f64(0.0f);
    svfloat64_t sum3 = svdup_f64(0.0f);

    // Process vectors in chunks, with unrolling for better pipelining
    size_t i = 0;
    for (; i + 4 * vl <= dimension; i += 4 * vl) {
        // Prefetch future data (critical for high dimensions)
        svprfw(svptrue_b64(), pVect1 + i + 16 * vl, SV_PLDL1KEEP);
        svprfw(svptrue_b64(), pVect2 + i + 16 * vl, SV_PLDL1KEEP);

        double *vec1_0 = pVect1 + i;
        double *vec2_0 = pVect2 + i;
        InnerProductStep(vec1_0, vec2_0, sum0);

        double *vec1_1 = pVect1 + i + vl;
        double *vec2_1 = pVect2 + i + vl;
        InnerProductStep(vec1_1, vec2_1, sum1);

        double *vec1_2 = pVect1 + i + 2 * vl;
        double *vec2_2 = pVect2 + i + 2 * vl;
        InnerProductStep(vec1_2, vec2_2, sum2);

        double *vec1_3 = pVect1 + i + 3 * vl;
        double *vec2_3 = pVect2 + i + 3 * vl;
        InnerProductStep(vec1_3, vec2_3, sum3);
    }

    // Handle remaining elements (less than 4*vl)
    for (; i < dimension; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dimension);
        svfloat64_t v1 = svld1_f64(pg, pVect1 + i);
        svfloat64_t v2 = svld1_f64(pg, pVect2 + i);
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