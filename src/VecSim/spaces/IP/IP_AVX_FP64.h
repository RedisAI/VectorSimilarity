/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"

static inline void InnerProductStep(double *&pVect1, double *&pVect2, __m256d &sum256) {
    __m256d v1 = _mm256_loadu_pd(pVect1);
    pVect1 += 4;
    __m256d v2 = _mm256_loadu_pd(pVect2);
    pVect2 += 4;
    sum256 = _mm256_add_pd(sum256, _mm256_mul_pd(v1, v2));
}

template <unsigned char residual> // 0..7
double FP64_InnerProductSIMD8Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty;

    __m256d sum256 = _mm256_setzero_pd();

    if (residual % 4) {
        // _mm256_maskz_loadu_pd is not available in AVX
        __mmask8 constexpr mask = (1 << (residual % 4)) - 1;
        __m256d v1 = my_mm256_maskz_loadu_pd<mask>(pVect1);
        pVect1 += residual % 4;
        __m256d v2 = my_mm256_maskz_loadu_pd<mask>(pVect2);
        pVect2 += residual % 4;
        sum256 = _mm256_mul_pd(v1, v2);
    }

    if (residual >= 4) {
        InnerProductStep(pVect1, pVect2, sum256);
    }

    do {
        InnerProductStep(pVect1, pVect2, sum256);
        InnerProductStep(pVect1, pVect2, sum256);
    } while (pVect1 < pEnd1);

    double PORTABLE_ALIGN32 TmpRes[4];
    _mm256_store_pd(TmpRes, sum256);
    double sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return 1.0 - sum;
}
