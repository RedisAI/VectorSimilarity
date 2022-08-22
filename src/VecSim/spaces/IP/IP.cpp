#include "IP.h"

float FP32_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *)pVect1)[i] * ((float *)pVect2)[i];
    }
    return res;
}

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t qty) {
    return 1.0f - FP32_InnerProduct_impl(pVect1, pVect2, qty);
}
