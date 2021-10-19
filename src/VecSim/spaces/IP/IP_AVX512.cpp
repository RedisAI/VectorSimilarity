
#include "IP_AVX512.h"
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP/IP.h"

#include <stdlib.h>

#ifndef __clang__

float InnerProductSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *)pVect1v;
    float *pVect2 = (float *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return 1.0f - sum;
}

float InnerProductSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                            const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext_AVX512(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *)pVect1v + qty16;
    float *pVect2 = (float *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return res + res_tail - 1.0f;
}

#else 

float InnerProductSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProductSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);


float InnerProductSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return InnerProductSIMD16ExtResiduals_AVX(pVect1v, pVect2v, qty_ptr);
}

float InnerProductSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return InnerProductSIMD4ExtResiduals_AVX(pVect1v, pVect2v, qty_ptr);
}

#endif // __clang__
