/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"
#include <cmath>


static inline void powerStep(float *&pVect1,__m256 &sumPowerReg) {

    __m256 v1 = _mm256_loadu_ps(pVect1);

    pVect1 += 8; 

    sumPowerReg = _mm256_add_ps(sumPowerReg,_mm256_mul_ps(v1, v1));

}

static inline void divStep(float *&pVect1,__m256 &normFactor) {

    __m256 v1 = _mm256_loadu_ps(pVect1);
    _mm256_storeu_ps(pVect1,_mm256_div_ps (v1,normFactor));

    pVect1 += 8; 

}


// normalize inplace
//Preformed in 2 steps:
//Power step - calculates the sum of powers 
//Div step - After sqrt sum of power, divides each chunk by norm factor/
//           and loads it into the vector (operation is inplace)
template <unsigned char residual> // 0..15
void FP32_normIMD16_AVX(const void *pVect1v,size_t dimension) {
    float *pVect1 = (float *)pVect1v;
    float *pVectOrig = pVect1;
    const float *pEnd1 = pVect1 + dimension;

    __m256 sumPowerReg = _mm256_setzero_ps();

    // Deal with 1-7 floats with mask loading, if needed
    if constexpr (residual % 8) {
        __mmask8 constexpr mask8 = (1 << (residual % 8)) - 1;
        __m256 v1 = my_mm256_maskz_loadu_ps<mask8>(pVect1);
        pVect1 += residual % 8;
        sumPowerReg = _mm256_mul_ps(v1, v1);
    }

    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 8) {
        powerStep(pVect1,sumPowerReg);

    }

    // We dealt with the residual part. We are left with some multiple of 16 floats.
    // In each iteration we calculate 16 floats = 512 bits.
    do {
        powerStep(pVect1, sumPowerReg);
        powerStep(pVect1, sumPowerReg);     
     
        // can reduce now or in the end of the loop, check both benchmark   
    } while (pVect1 < pEnd1);


    pVect1 = pVectOrig;

    float sumOfPower = my_mm256_reduce_add_ps(sumPowerReg);
    __m256 normFactor = _mm256_sqrt_ps(_mm256_set1_ps(sumOfPower));


    // Deal with 1-7 floats with mask loading, if needed
    if constexpr (residual % 8) {
        __mmask8 constexpr mask8 = (1 << (residual % 8)) - 1;
        __m256 v1 = _mm256_loadu_ps(pVect1);
        __m256 blend = _mm256_blend_ps(v1,_mm256_div_ps(v1,normFactor),mask8);
        _mm256_storeu_ps(pVect1,blend);

        pVect1 += residual % 8;
    }
    // If the reminder is >=8, have another step of 8 floats
    if constexpr (residual >= 8) {
        divStep(pVect1,normFactor);
    }

    do {
        divStep(pVect1, normFactor);     
        divStep(pVect1, normFactor);     

        // can reduce now or in the end of the loop, check both benchmark   
    } while (pVect1 < pEnd1);

}