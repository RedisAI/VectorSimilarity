
#include "IP.h"

#include <stdlib.h>

float FP32_InnerProduct_impl(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *)pVect1)[i] * ((float *)pVect2)[i];
    }
    return res;
}

float FP32_InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - FP32_InnerProduct_impl(pVect1, pVect2, qty_ptr);
}
