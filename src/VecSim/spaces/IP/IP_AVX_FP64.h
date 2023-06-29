/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"

static inline void InnerProductStep(double *&pVect1, double *&pVect2, __m256d &sum256) {
    __m256d v1 = _mm256_loadu_pd(pVect1);
    pVect1 += 4;
    __m256d v2 = _mm256_loadu_pd(pVect2);
    pVect2 += 4;
    sum256 = _mm256_add_pd(sum256, _mm256_mul_pd(v1, v2));
}

// TODO: verify if this is the correct way to implement this function
template <__mmask8 mask>
static inline __m256d my_mm256_maskz_loadu_pd(const double *p) {
    __m128i indices = _mm_set_epi32(3, 2, 1, 0); // Set the indices for loading 4 double values
    __m256d vec_mask = _mm256_blend_pd(_mm256_set1_pd(-1), _mm256_setzero_pd(), mask); // Set the mask for loading 8 float values (1 if mask is true, 0 if mask is false

    __m256d loaded_values = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), p, indices, vec_mask, 4);

    return loaded_values;
}

template <__mmask8 mask>
double FP64_InnerProductSIMD8Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty - 8;

    __m256d sum256 = _mm256_setzero_pd();

    while (pVect1 <= pEnd1) {
        InnerProductStep(pVect1, pVect2, sum256);
        InnerProductStep(pVect1, pVect2, sum256);
    }

    if (mask >= 0x0F) {
        InnerProductStep(pVect1, pVect2, sum256);
    }

    // _mm256_maskz_loadu_pd is not available in AVX
    __mmask8 constexpr mask4 = mask & (mask >> 4);
    if (mask4 != 0) {
        __m256d v1 = my_mm256_maskz_loadu_pd<mask4>(pVect1);
        __m256d v2 = my_mm256_maskz_loadu_pd<mask4>(pVect2);
        sum256 = _mm256_add_pd(sum256, _mm256_mul_pd(v1, v2));
    }

    double PORTABLE_ALIGN32 TmpRes[4];
    _mm256_store_pd(TmpRes, sum256);
    double sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return 1.0 - sum;
}
