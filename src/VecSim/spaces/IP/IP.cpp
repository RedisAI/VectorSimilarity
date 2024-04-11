/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "IP.h"
#include "VecSim/utils/types_decl.h"
#include <cstring>

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    float *vec1 = (float *)pVect1;
    float *vec2 = (float *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += vec1[i] * vec2[i];
    }
    return 1.0f - res;
}

double FP64_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    double *vec1 = (double *)pVect1;
    double *vec2 = (double *)pVect2;

    double res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += vec1[i] * vec2[i];
    }
    return 1.0 - res;
}

float BF16_InnerProduct_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = 0;
        float b = 0;
        memcpy((bfloat16 *)&a + 1, pVect1 + i, sizeof(bfloat16));
        memcpy((bfloat16 *)&b + 1, pVect2 + i, sizeof(bfloat16));
        res += a * b;
    }
    return 1.0 - res;
}

float BF16_InnerProduct_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = 0;
        float b = 0;
        memcpy(&a, pVect1 + i, sizeof(bfloat16));
        memcpy(&b, pVect2 + i, sizeof(bfloat16));
        res += a * b;
    }
    return 1.0 - res;
}
