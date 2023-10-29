/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "L2.h"

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
