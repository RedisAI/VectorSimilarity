/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "IP.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include <cstring>

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

float FLOAT_INTEGER_InnerProduct(const float *pVect1v, const uint8_t *pVect2v, size_t dimension,
                                 float min_val, float delta) {
    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float dequantized_V2 = (pVect2v[i] * delta + min_val);
        res += pVect1v[i] * dequantized_V2;
    }
    return res;
}

float SQ8_InnerProduct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const float *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);
    // pVect2 is a vector of uint8_t, so we need to de-quantize it, normalize it and then multiply
    // it. it is structured as [quantized values (int8_t * dim)][min_val (float)][delta
    // (float)]] The last two values are used to dequantize the vector.
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    // Compute inner product with dequantization
    const float res = FLOAT_INTEGER_InnerProduct(pVect1, pVect2, dimension, min_val, delta);
    return 1.0f - res;
}

float SQ8_Cosine(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const float *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get quantization parameters
    const float min_val = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    // Compute inner product with dequantization
    const float res = FLOAT_INTEGER_InnerProduct(pVect1, pVect2, dimension, min_val, delta);
    return 1.0f - res;
}

// SQ8-to-SQ8: Both vectors are uint8 quantized
float SQ8_SQ8_InnerProduct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get quantization parameters from pVect1
    const float min_val1 = *reinterpret_cast<const float *>(pVect1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVect1 + dimension + sizeof(float));

    // Get quantization parameters from pVect2
    const float min_val2 = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));

    // Compute inner product with dequantization of both vectors
    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float dequant1 = pVect1[i] * delta1 + min_val1;
        float dequant2 = pVect2[i] * delta2 + min_val2;
        res += dequant1 * dequant2;
    }
    return 1.0f - res;
}

// SQ8-to-SQ8: Both vectors are uint8 quantized
float SQ8_SQ8_InnerProduct_Precomputed(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get quantization parameters from pVect1
    const float min_val1 = *reinterpret_cast<const float *>(pVect1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVect1 + dimension + sizeof(float));
    const float sum1 = *reinterpret_cast<const float *>(pVect1 + dimension + 2 * sizeof(float));

    // Get quantization parameters from pVect2
    const float min_val2 = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    const float sum2 = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));

    // Compute inner product with dequantization of both vectors
    float product = 0;
    for (size_t i = 0; i < dimension; i++) {
        product += pVect1[i] * pVect2[i];
    }
    float res = min_val1 * sum2 + min_val2 * sum1 - dimension * min_val1 * min_val2 +
               delta1 * delta2 * product;
    return 1.0f - res;
}

// SQ8-to-SQ8: Both vectors are uint8 quantized (cosine version)
float SQ8_SQ8_Cosine(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get quantization parameters from pVect1
    const float min_val1 = *reinterpret_cast<const float *>(pVect1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVect1 + dimension + sizeof(float));

    // Get quantization parameters from pVect2
    const float min_val2 = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));

    // Compute inner product with dequantization of both vectors
    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        float dequant1 = pVect1[i] * delta1 + min_val1;
        float dequant2 = pVect2[i] * delta2 + min_val2;
        res += dequant1 * dequant2;
    }
    // Assume both vectors are normalized.
    return 1.0f - res;
}

// SQ8-to-SQ8: Both vectors are uint8 quantized (cosine version)
float SQ8_SQ8_Cosine_Precomputed(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);

    // Get quantization parameters from pVect1
    const float min_val1 = *reinterpret_cast<const float *>(pVect1 + dimension);
    const float delta1 = *reinterpret_cast<const float *>(pVect1 + dimension + sizeof(float));
    const float sum1 = *reinterpret_cast<const float *>(pVect1 + dimension + 2 * sizeof(float));

    // Get quantization parameters from pVect2
    const float min_val2 = *reinterpret_cast<const float *>(pVect2 + dimension);
    const float delta2 = *reinterpret_cast<const float *>(pVect2 + dimension + sizeof(float));
    const float sum2 = *reinterpret_cast<const float *>(pVect2 + dimension + 2 * sizeof(float));

    float product = 0;
    for (size_t i = 0; i < dimension; i++) {
        product += pVect1[i] * pVect2[i];
    }

    float res = min_val1 * sum2 + min_val2 * sum1 - dimension * min_val1 * min_val2 +
               delta1 * delta2 * product;
    return 1.0f - res;
}

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

// Return type for the inner product functions.
// The type should be able to hold `dimension * MAX_VAL(int_elem_t) * MAX_VAL(int_elem_t)`.
// To support dimension up to 2^16, we need the difference between the type and int_elem_t to be at
// least 2 bytes. We assert that in the implementation.
template <typename int_elem_t>
using ret_t = std::conditional_t<sizeof(int_elem_t) == 1, int, long long>;

template <typename int_elem_t>
static inline ret_t<int_elem_t>
INTEGER_InnerProductImp(const int_elem_t *pVect1, const int_elem_t *pVect2, size_t dimension) {
    static_assert(sizeof(ret_t<int_elem_t>) - sizeof(int_elem_t) * 2 >= sizeof(uint16_t));
    ret_t<int_elem_t> res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += pVect1[i] * pVect2[i];
    }
    return res;
}

float INT8_InnerProduct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const int8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const int8_t *>(pVect2v);
    return 1 - INTEGER_InnerProductImp(pVect1, pVect2, dimension);
}

float INT8_Cosine(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const int8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const int8_t *>(pVect2v);
    // We expect the vectors' norm to be stored at the end of the vector.
    float norm_v1 = *reinterpret_cast<const float *>(pVect1 + dimension);
    float norm_v2 = *reinterpret_cast<const float *>(pVect2 + dimension);
    return 1.0f - float(INTEGER_InnerProductImp(pVect1, pVect2, dimension)) / (norm_v1 * norm_v2);
}

float UINT8_InnerProduct(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);
    return 1 - INTEGER_InnerProductImp(pVect1, pVect2, dimension);
}

float UINT8_Cosine(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const auto *pVect1 = static_cast<const uint8_t *>(pVect1v);
    const auto *pVect2 = static_cast<const uint8_t *>(pVect2v);
    // We expect the vectors' norm to be stored at the end of the vector.
    float norm_v1 = *reinterpret_cast<const float *>(pVect1 + dimension);
    float norm_v2 = *reinterpret_cast<const float *>(pVect2 + dimension);
    return 1.0f - float(INTEGER_InnerProductImp(pVect1, pVect2, dimension)) / (norm_v1 * norm_v2);
}
