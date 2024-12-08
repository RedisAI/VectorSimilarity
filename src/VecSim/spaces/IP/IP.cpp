/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "IP.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include <cstring>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

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

template <bool is_little>
float BF16_InnerProduct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *pVect1 = (bfloat16 *)pVect1v;
    auto *pVect2 = (bfloat16 *)pVect2v;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float a = vecsim_types::bfloat16_to_float32<is_little>(pVect1[i]);
        float b = vecsim_types::bfloat16_to_float32<is_little>(pVect2[i]);
        res += a * b;
    }
    return 1.0f - res;
}

float BF16_InnerProduct_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return BF16_InnerProduct<true>(pVect1v, pVect2v, dimension);
}

float BF16_InnerProduct_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return BF16_InnerProduct<false>(pVect1v, pVect2v, dimension);
}

float FP16_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (float16 *)pVect1;
    auto *vec2 = (float16 *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += vecsim_types::FP16_to_FP32(vec1[i]) * vecsim_types::FP16_to_FP32(vec2[i]);
    }
    return 1.0f - res;
}

static inline int INT8_InnerProductImp(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    int res = 0;
    for (size_t i = 0; i < dimension; i++) {
        int16_t a = pVect1[i];
        int16_t b = pVect2[i];
        res += a * b;
    }
    return res;
}

float INT8_InnerProduct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    return 1 - INT8_InnerProductImp(pVect1v, pVect2v, dimension);
}

float INT8_Cosine(const void *pVect1v, const void *pVect2v, size_t dimension) {
    float norm_v1 = *(float *)((int8_t *)pVect1v + dimension);
    float norm_v2 = *(float *)((int8_t *)pVect2v + dimension);
    return 1.0f - float(INT8_InnerProductImp(pVect1v, pVect2v, dimension)) / (norm_v1 * norm_v2);
}
