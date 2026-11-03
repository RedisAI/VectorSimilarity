/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/types/sq8.h"

namespace vecsim_types {

// Two scalar-quantized coordinates per byte, followed by the same metadata as SQ8.
struct sq4 : sq8 {
    static constexpr uint8_t max_code = 15;

    static constexpr size_t data_bytes_count(size_t dim) { return dim / 2 + dim % 2; }

    static uint8_t load_code(const uint8_t *data, size_t index) {
        return (data[index / 2] >> ((index % 2) * 4)) & max_code;
    }

    // Coordinates are stored in ascending order. Starting each byte with an assignment allows
    // in-place preprocessing and leaves the unused high nibble zero for odd dimensions.
    static void store_code(uint8_t *data, size_t index, uint8_t code) {
        if (index % 2 == 0) {
            data[index / 2] = code & max_code;
        } else {
            data[index / 2] = (data[index / 2] & max_code) | ((code & max_code) << 4);
        }
    }

    template <VecSimMetric Metric, bool WithNorm = false>
    static constexpr size_t storage_bytes_count(size_t dim) {
        return data_bytes_count(dim) + storage_metadata_count<Metric, WithNorm>() * sizeof(float);
    }
};

} // namespace vecsim_types
