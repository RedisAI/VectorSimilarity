/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "L2.h"
#include "VecSim/types/bfloat16.h"
#include <cstring>

using bfloat16 = vecsim_types::bfloat16;

float FP32_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float *vec1 = (float *)pVect1v;
    float *vec2 = (float *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float t = vec1[i] - vec2[i];
        res += t * t;
    }
    return res;
}

double FP64_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    double *vec1 = (double *)pVect1v;
    double *vec2 = (double *)pVect2v;

    double res = 0;
    for (size_t i = 0; i < dimension; i++) {
        double t = vec1[i] - vec2[i];
        res += t * t;
    }
    return res;
}

float BF16_L2Sqr_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = 0;
        float b = 0;
        memcpy((bfloat16 *)&a + 1, pVect1 + i, sizeof(bfloat16));
        memcpy((bfloat16 *)&b + 1, pVect2 + i, sizeof(bfloat16));
        float diff = a - b;
        res += diff * diff;
    }
    return res;
}

float BF16_L2Sqr_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = 0;
        float b = 0;
        memcpy(&a, pVect1 + i, sizeof(bfloat16));
        memcpy(&b, pVect2 + i, sizeof(bfloat16));
        float diff = a - b;
        res += diff * diff;
    }
    return res;
}
