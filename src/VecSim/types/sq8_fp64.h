/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cstdint>
#include <limits>
#include "VecSim/types/quantized_metadata.h"

namespace vecsim_types {

// SQ8 bytes followed by double reconstruction metadata; distinct from the FP32 SQ8 layout.
struct sq8_fp64 : quantized_metadata {
    using value_type = uint8_t;
    static constexpr uint8_t max_code = std::numeric_limits<uint8_t>::max();
    static constexpr size_t data_bytes_count(size_t dim);
    template <VecSimMetric Metric>
    static constexpr size_t storage_bytes_count(size_t dim);
    template <VecSimMetric Metric>
    static constexpr size_t query_bytes_count(size_t dim);
};

constexpr size_t sq8_fp64::data_bytes_count(size_t dim) { return dim; }

template <VecSimMetric Metric>
constexpr size_t sq8_fp64::storage_bytes_count(size_t dim) {
    return data_bytes_count(dim) + storage_metadata_count<Metric>() * sizeof(double);
}

template <VecSimMetric Metric>
constexpr size_t sq8_fp64::query_bytes_count(size_t dim) {
    return (dim + query_metadata_count<Metric>()) * sizeof(double);
}

} // namespace vecsim_types
