/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace vecsim_types {

using bfloat16 = unsigned short;

static inline bfloat16 float_to_bf16(const float ff) {
    uint32_t f32 = 0;
    memcpy(&f32, &ff, sizeof(ff));
    uint32_t lsb = (f32 >> 16) & 1;
    uint32_t round = lsb + 0x7FFF;
    f32 += round;
    return f32 >> 16;
}

static inline float bfloat16_to_float32(bfloat16 val) {
    float result = 0;
    memcpy((bfloat16 *)&result + 1, &val, sizeof(bfloat16));
    return result;
}

static inline float bfloat16_to_float32_bigEndian(bfloat16 val) {
    float result = 0;
    memcpy((bfloat16 *)&result, &val, sizeof(bfloat16));
    return result;
}

} // namespace vecsim_types
