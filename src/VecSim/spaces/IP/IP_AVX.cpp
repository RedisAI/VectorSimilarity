#include "IP_AVX.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP.h"

float FP32_InnerProductSIMD16Ext_AVX_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    size_t qty16 = qty >> 4 << 4;

    const float *pEnd1 = pVect1 + qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7];

    return sum;
}

float FP32_InnerProductSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    return 1.0f - FP32_InnerProductSIMD16Ext_AVX_impl(pVect1v, pVect2v, qty);
}

float FP32_InnerProductSIMD4Ext_AVX_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    float PORTABLE_ALIGN16 TmpRes[4];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    size_t qty16 = qty >> 4 << 4;
    size_t qty4 = qty >> 2 << 2;

    const float *pEnd1 = pVect1 + qty16;
    const float *pEnd2 = pVect1 + qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod =
        _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float FP32_InnerProductSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    return 1.0f - FP32_InnerProductSIMD4Ext_AVX_impl(pVect1v, pVect2v, qty);
}

float FP32_InnerProductSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v,
                                              size_t qty) {
    size_t qty16 = qty >> 4 << 4;
    float res = FP32_InnerProductSIMD16Ext_AVX_impl(pVect1v, pVect2v, qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = FP32_InnerProduct_impl(pVect1, pVect2, qty_left);
    return 1.0f - (res + res_tail);
}

float FP32_InnerProductSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty) {
    size_t qty4 = qty >> 2 << 2;

    float res = FP32_InnerProductSIMD4Ext_AVX_impl(pVect1v, pVect2v, qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *)pVect1v + qty4;
    float *pVect2 = (float *)pVect2v + qty4;
    float res_tail = FP32_InnerProduct_impl(pVect1, pVect2, qty_left);

    return 1.0f - (res + res_tail);
}
