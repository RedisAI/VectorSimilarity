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

template <bool is_little = true>
inline int16_t int8_to_int16(int8_t val) {
    size_t constexpr bytes_offset = is_little ? 1 : 0;
    int result = 0;
    int16_t *p_result = (int16_t *)&result + bytes_offset;
    *p_result = val;
    return result;
}

} // namespace vecsim_types
