/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <cstdint>
#include "IP.h"

inline float _interpret_as_float(uint32_t num) {
    void *num_ptr = &num;
    return *(float *)num_ptr;
}

inline int32_t _interpret_as_int(float num) {
    void *num_ptr = &num;
    return *(int32_t *)num_ptr;
}

float FP16_to_FP32(uint16_t input) {
    // https://gist.github.com/2144712
    // Fabian "ryg" Giesen.

    const uint32_t shifted_exp = 0x7c00u << 13; // exponent mask after shift

    int32_t o = ((int32_t)(input & 0x7fffu)) << 13; // exponent/mantissa bits
    int32_t exp = shifted_exp & o;              // just the exponent
    o += (int32_t)(127 - 15) << 23;             // exponent adjust

    int32_t infnan_val = o + ((int32_t)(128 - 16) << 23);
    int32_t zerodenorm_val =
        _interpret_as_int(_interpret_as_float(o + (1u << 23)) - _interpret_as_float(113u << 23));
    int32_t reg_val = (exp == 0) ? zerodenorm_val : o;

    int32_t sign_bit = ((int32_t)(input & 0x8000u)) << 16;
    return _interpret_as_float(((exp == shifted_exp) ? infnan_val : reg_val) | sign_bit);
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

float FP16_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension) {
    auto *vec1 = (uint16_t *)pVect1;
    auto *vec2 = (uint16_t *)pVect2;

    float res = 0;
    for (size_t i = 0; i < dimension; i++) {
        res += FP16_to_FP32(vec1[i]) * FP16_to_FP32(vec2[i]);
    }
    return 1.0f - res;
}
