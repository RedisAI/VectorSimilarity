#include "IP.h"

float FP32_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    float *vec1_ptr = (float *)pVect1;
    float *vec2_ptr = (float *)pVect2;  

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += vec1_ptr[i] * vec2_ptr[i];
    }
    return res;
}

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t qty) {
    return 1.0f - FP32_InnerProduct_impl(pVect1, pVect2, qty);
}

double FP64_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    double *vec1_ptr = (double *)pVect1;
    double *vec2_ptr = (double *)pVect2;    
    
    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += vec1_ptr[i] * vec2_ptr[i];
    }
    return res;
}

double FP64_InnerProduct(const void *pVect1, const void *pVect2, size_t qty) {
    return 1.0 - FP64_InnerProduct_impl(pVect1, pVect2, qty);
}
