/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "IP.h"
#include <cstring>

float BFP16_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    u_int16_t *vec1 = (u_int16_t *)pVect1;
    u_int16_t *vec2 = (u_int16_t *)pVect2;

    float res = 0;
    float a = 0;
    float b = 0;
    // verify all zeros
    memset(&a, 0, sizeof(float));
    memset(&b, 0, sizeof(float));
    for (size_t i = 0; i < qty; i++) {
        u_int16_t a16 = vec1[i];
        u_int16_t b16 = vec2[i];
        memcpy(&a, &a16, sizeof(u_int16_t));
        memcpy(&b, &b16, sizeof(u_int16_t));
        res += a * b;
    }
    return res;
}

float BFP16_InnerProduct(const void *pVect1, const void *pVect2, size_t qty)  {
    return 1.0f - BFP16_InnerProduct_impl(pVect1, pVect2, qty);
}

float FP32_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    float *vec1 = (float *)pVect1;
    float *vec2 = (float *)pVect2;

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t qty) {
    return 1.0f - FP32_InnerProduct_impl(pVect1, pVect2, qty);
}

double FP64_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    double *vec1 = (double *)pVect1;
    double *vec2 = (double *)pVect2;

    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

double FP64_InnerProduct(const void *pVect1, const void *pVect2, size_t qty) {
    return 1.0 - FP64_InnerProduct_impl(pVect1, pVect2, qty);
}
