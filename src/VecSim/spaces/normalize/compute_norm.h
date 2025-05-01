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
#include <type_traits>

namespace spaces {

template <typename DataType>
static inline float IntegralType_ComputeNorm(const DataType *vec, const size_t dim) {
    static_assert(std::is_integral_v<DataType>, "DataType must be an integral type");

    int sum = 0;

    for (size_t i = 0; i < dim; i++) {
        // No need to cast to int because c++ integer promotion ensures vec[i] is promoted to int
        // before multiplication.
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

} // namespace spaces
