/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <cstdint>
#include "IP.h"
#include "VecSim/types/float16.h"

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (float *)pVect1;
    auto *vec2 = (float *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += vec1[i] * vec2[i];
    }
    return 1.0f - res;
}

double FP64_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (double *)pVect1;
    auto *vec2 = (double *)pVect2;

    double res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += vec1[i] * vec2[i];
    }
    return 1.0 - res;
}

float FP16_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (float16 *)pVect1;
    auto *vec2 = (float16 *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += FP16_to_FP32(vec1[i]) * FP16_to_FP32(vec2[i]);
    }
    return 1.0f - res;
}
