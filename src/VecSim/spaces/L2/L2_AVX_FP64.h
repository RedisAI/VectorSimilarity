/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/AVX_utils.h"

static inline void L2SqrStep(double *&pVect1, double *&pVect2, __m256d &sum) {
    __m256d v1 = _mm256_loadu_pd(pVect1);
    pVect1 += 4;
    __m256d v2 = _mm256_loadu_pd(pVect2);
    pVect2 += 4;
    __m256d diff = _mm256_sub_pd(v1, v2);
    // sum = _mm256_fmadd_pd(diff, diff, sum);
    sum = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
}

template <unsigned char residual> // 0..7
double FP64_L2SqrSIMD8Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    double *pVect1 = (double *)pVect1v;
    double *pVect2 = (double *)pVect2v;

    const double *pEnd1 = pVect1 + qty;

    __m256d sum = _mm256_setzero_pd();

    // _mm256_maskz_loadu_pd is not available in AVX
    if (residual % 4) {
        __mmask8 constexpr mask4 = (1 << (residual % 4)) - 1;
        __m256d v1 = my_mm256_maskz_loadu_pd<mask4>(pVect1);
        pVect1 += residual % 4;
        __m256d v2 = my_mm256_maskz_loadu_pd<mask4>(pVect2);
        pVect2 += residual % 4;
        __m256d diff = _mm256_sub_pd(v1, v2);
        sum = _mm256_mul_pd(diff, diff);
    }

    if (residual >= 4) {
        L2SqrStep(pVect1, pVect2, sum);
    }

    while (pVect1 < pEnd1) {
        L2SqrStep(pVect1, pVect2, sum);
        L2SqrStep(pVect1, pVect2, sum);
    }

    double PORTABLE_ALIGN32 TmpRes[4];
    _mm256_store_pd(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
