/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "space_includes.h"

template <__mmask8 mask> // 1..7
static inline __m256 my_mm256_maskz_loadu_ps(const float *p) {
    __m256 data = _mm256_loadu_ps(p);
    // Set the mask for loading 8 float values (1 if mask is true, 0 if mask is false
    __m256 masked_data = _mm256_blend_ps(_mm256_setzero_ps(), data, mask);

    return masked_data;
}

template <__mmask8 mask> // 1..3
static inline __m256d my_mm256_maskz_loadu_pd(const double *p) {
    __m256d data = _mm256_loadu_pd(p);
    // Set the mask for loading 8 float values (1 if mask is true, 0 if mask is false
    __m256d masked_data = _mm256_blend_pd(_mm256_setzero_pd(), data, mask);

    return masked_data;
}