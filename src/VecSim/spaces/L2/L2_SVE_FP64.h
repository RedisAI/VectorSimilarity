/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <arm_sve.h>

static void L2SquareStep(double *&pVect1, double *&pVect2, svfloat64_t &sum) {
    // Load vectors
    svfloat64_t v1 = svld1_f64(svptrue_b64(), pVect1);
    svfloat64_t v2 = svld1_f64(svptrue_b64(), pVect2);

    // Calculate difference between vectors
    svfloat64_t diff = svsub_f64_z(svptrue_b64(), v1, v2);

    // Square the difference and accumulate: sum += diff * diff
    sum = svmla_f64_z(svptrue_b64(), sum, diff, diff);

    // Advance pointers by the vector length
    pVect1 += svcntw();
    pVect2 += svcntw();
}

template <unsigned char residual>
double FP64_L2SqrSIMD_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const double *pVect1 = (double*) pVect1v;
    const double *pVect2 = static_cast<const double *>(pVect2v);

    // Get the number of 64-bit elements per vector at runtime
    uint64_t vl = svcntw();

    // Multiple accumulators to increase instruction-level parallelism
    svfloat64_t sum0 = svdup_f64(0.0f);
    svfloat64_t sum1 = svdup_f64(0.0f);
    svfloat64_t sum2 = svdup_f64(0.0f);
    svfloat64_t sum3 = svdup_f64(0.0f);

    // Process vectors in chunks, with unrolling for better pipelining
    size_t i = 0;
    for (; i + 4 * vl <= dimension; i += 4 * vl) {
        // Prefetch future data
        svprfw(svptrue_b64(), pVect1 + i + 16 * vl, SV_PLDL1KEEP);
        svprfw(svptrue_b64(), pVect2 + i + 16 * vl, SV_PLDL1KEEP);

        // Process 4 chunks with separate accumulators
        double *vec1_0 = pVect1 + i;
        double *vec2_0 = pVect2 + i;
        L2SquareStep(vec1_0, vec2_0, sum0);
        
        double *vec1_1 = pVect1 + i + vl;
        double *vec2_1 = pVect2 + i + vl;
        L2SquareStep(vec1_1, vec2_1, sum1);
        
        double *vec1_2 = pVect1 + i + 2 * vl;
        double *vec2_2 = pVect2 + i + 2 * vl;
        L2SquareStep(vec1_2, vec2_2, sum2);
        
        double *vec1_3 = pVect1 + i + 3 * vl;
        double *vec2_3 = pVect2 + i + 3 * vl;
        L2SquareStep(vec1_3, vec2_3, sum3);

    }

    // Handle remaining elements (less than 4*vl)
    for (; i < dimension; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dimension);

        // Load vectors with predication
        svfloat64_t v1 = svld1_f64(pg, pVect1 + i);
        svfloat64_t v2 = svld1_f64(pg, pVect2 + i);

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