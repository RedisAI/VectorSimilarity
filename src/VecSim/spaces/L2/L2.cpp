/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "L2.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include <cstring>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

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

template <bool is_little>
float BF16_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension) {
    bfloat16 *pVect1 = (bfloat16 *)pVect1v;
    bfloat16 *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = vecsim_types::bfloat16_to_float32<is_little>(pVect1[i]);
        float b = vecsim_types::bfloat16_to_float32<is_little>(pVect2[i]);
        float diff = a - b;
        res += diff * diff;
    }
    return res;
}

float BF16_L2Sqr_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return BF16_L2Sqr<true>(pVect1v, pVect2v, dimension);
}

float BF16_L2Sqr_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return BF16_L2Sqr<false>(pVect1v, pVect2v, dimension);
}

float FP16_L2Sqr(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (float16 *)pVect1;
    auto *vec2 = (float16 *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float t = vecsim_types::FP16_to_FP32(vec1[i]) - vecsim_types::FP16_to_FP32(vec2[i]);
        res += t * t;
    }
    return res;
}
