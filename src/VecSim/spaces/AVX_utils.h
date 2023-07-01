/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "space_includes.h"

template <__mmask8 mask>
static inline __m256 my_mm256_maskz_loadu_ps(const float *p) {
    // Set the indices for loading 8 float values
    __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    // Set the mask for loading 8 float values (1 if mask is true, 0 if mask is false
    __m256 vec_mask = _mm256_blend_ps(_mm256_setzero_ps(), _mm256_set1_ps(-1), mask);

    __m256 loaded_values = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), p, indices, vec_mask, 4);

    return loaded_values;
}

template <__mmask8 mask>
static inline __m256d my_mm256_maskz_loadu_pd(const double *p) {
    // Set the indices for loading 4 double values
    __m128i indices = _mm_set_epi32(3, 2, 1, 0);
    // Set the mask for loading 8 float values (1 if mask is true, 0 if mask is false
    __m256d vec_mask = _mm256_blend_pd(_mm256_setzero_pd(), _mm256_set1_pd(-1), mask);

    __m256d loaded_values = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), p, indices, vec_mask, 8);

    return loaded_values;
}
