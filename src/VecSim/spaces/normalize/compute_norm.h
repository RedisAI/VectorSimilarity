/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace spaces {

template <typename DataType>
static inline float IntegralType_ComputeNorm(const DataType *vec, const size_t dim) {
    static_assert(std::is_integral_v<DataType>, "DataType must be an integral type");

    // uint64_t, not int. Each uint8 term reaches 255*255 = 65,025, so the total passes INT32_MAX
    // from dimension 33,026 and the accumulation was signed-overflow UB there. This is the norm the
    // cosine preprocessor writes for every stored vector and every query, so a wrong value here
    // reaches the kernels before they run: at dimension 65,537 the norm came out NaN, and at 66,052
    // it came out 252.99 instead of about 65,536.49. int8 has the same shape with a higher bound,
    // 16,129 per term, so it overflowed from dimension 133,153.
    uint64_t sum = 0;

    for (size_t i = 0; i < dim; i++) {
        // The element promotes to int before multiplying, which is exact for one- and two-byte
        // types; only the running total needs the wider type.
        const int64_t term = static_cast<int64_t>(vec[i]) * vec[i];
        sum += static_cast<uint64_t>(term);
    }
    return std::sqrt(static_cast<double>(sum));
}

} // namespace spaces
