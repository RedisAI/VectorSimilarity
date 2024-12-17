/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"

static inline void powerStep(double *&pVect1, __m256d &sumPowerReg) {

    __m256d v1 = _mm256_loadu_pd(pVect1);

    pVect1 += 4;

    sumPowerReg = _mm256_add_pd(sumPowerReg, _mm256_mul_pd(v1, v1));
}

static inline void divStep(double *&pVect1, __m256d &normFactor) {

    __m256d v1 = _mm256_loadu_pd(pVect1);
    _mm256_storeu_pd(pVect1, _mm256_div_pd(v1, normFactor));

    pVect1 += 4;
}

// normalize inplace
// Preformed in 2 steps:
// Power step - calculates the sum of powers
// Div step - After sqrt sum of power, divides each chunk by norm factor/
//           and loads it into the vector (operation is inplace)
template <unsigned char residual> // 0..15
static void FP64_normalizeSIMD8_AVX(void *pVect1v, size_t dimension) {
    double *pVect1 = (double *)pVect1v;
    const double *pEnd1 = pVect1 + dimension;

    __m256d sumPowerReg = _mm256_setzero_pd();

    // Deal with 1-7 floats with mask loading, if needed
    if constexpr (residual % 4) {
        __mmask8 constexpr mask4 = (1 << (residual % 4)) - 1;
        __m256d v1 = my_mm256_maskz_loadu_pd<mask4>(pVect1);
        pVect1 += residual % 4;
        sumPowerReg = _mm256_mul_pd(v1, v1);
    }

    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 4) {
        powerStep(pVect1, sumPowerReg);
    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        powerStep(pVect1, sumPowerReg);
        powerStep(pVect1, sumPowerReg);

        // can reduce now or in the end of the loop, check both benchmark
    } while (pVect1 < pEnd1);

    pVect1 = (double *)pVect1v;

    double sumOfPower = my_mm256_reduce_add_pd(sumPowerReg);
    __m256d normFactor = _mm256_sqrt_pd(_mm256_set1_pd(sumOfPower));

    // Deal with 1-7 floats with mask loading, if needed
    if constexpr (residual % 4) {
        __mmask8 constexpr mask4 = (1 << (residual % 4)) - 1;
        __m256d v1 = _mm256_loadu_pd(pVect1);
        __m256d blend = _mm256_blend_pd(v1, _mm256_div_pd(v1, normFactor), mask4);
        _mm256_storeu_pd(pVect1, blend);

        pVect1 += residual % 4;
    }
    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 4) {
        divStep(pVect1, normFactor);
    }

    do {
        divStep(pVect1, normFactor);
        divStep(pVect1, normFactor);

        // can reduce now or in the end of the loop, check both benchmark
    } while (pVect1 < pEnd1);
}