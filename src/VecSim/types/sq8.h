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
#include <cstddef>
#include <limits>
#include "VecSim/vec_sim_common.h"

namespace vecsim_types {

// Represents a scalar-quantized 8-bit blob with reconstruction metadata
struct sq8 {
    using value_type = uint8_t;
    static constexpr uint8_t max_code = std::numeric_limits<uint8_t>::max();

    static constexpr size_t data_bytes_count(size_t dim) { return dim; }

    static uint8_t load_code(const uint8_t *data, size_t index) { return data[index]; }

    static void store_code(uint8_t *data, size_t index, uint8_t code) { data[index] = code; }

    // Dot product of two code arrays: sum(q1_i * q2_i).
    static float code_dot(const uint8_t *a, const uint8_t *b, size_t dim) {
        float product = 0;
        for (size_t i = 0; i < dim; i++) {
            product += load_code(a, i) * load_code(b, i);
        }
        return product;
    }

    // Dot product of a code array with FP32 values: sum(q_i * y_i). Uses 4 accumulators for
    // instruction-level parallelism.
    static float code_dot_fp32(const uint8_t *codes, const float *y, size_t dim) {
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

        // Main loop: process 4 elements per iteration
        size_t i = 0;
        size_t dim4 = dim & ~size_t(3); // dim4 is a multiple of 4
        for (; i < dim4; i += 4) {
            sum0 += static_cast<float>(load_code(codes, i + 0)) * y[i + 0];
            sum1 += static_cast<float>(load_code(codes, i + 1)) * y[i + 1];
            sum2 += static_cast<float>(load_code(codes, i + 2)) * y[i + 2];
            sum3 += static_cast<float>(load_code(codes, i + 3)) * y[i + 3];
        }

        // Handle remainder (0-3 elements)
        for (; i < dim; i++) {
            sum0 += static_cast<float>(load_code(codes, i)) * y[i];
        }

        // Combine accumulators
        return (sum0 + sum1) + (sum2 + sum3);
    }

    // Metadata layout indices (stored after quantized values)
    enum MetadataIndex : size_t {
        MIN_VAL = 0,
        DELTA = 1,
        SUM = 2,
        SUM_SQUARES = 3 // Only for L2
    };

    enum QueryMetadataIndex : size_t {
        SUM_QUERY = 0,
        SUM_SQUARES_QUERY = 1 // Only for L2
    };

    // Template on Metric and WithNorm — compile-time constants
    // WithNorm: one extra metadata slot for x_mean_ip / y_mean_ip. Only needed for IP: the
    // normalized L2 correction cancels the mean terms exactly during centering and never reads
    // mean_ip (see DistanceCalculatorWithNorm), so no extra slot is allocated for L2.
    template <VecSimMetric Metric, bool WithNorm = false>
    static constexpr size_t storage_metadata_count() {
        return ((Metric == VecSimMetric_L2) ? 4 : 3) +
               ((WithNorm && Metric == VecSimMetric_IP) ? 1 : 0);
    }

    template <VecSimMetric Metric, bool WithNorm = false>
    static constexpr size_t query_metadata_count() {
        return ((Metric == VecSimMetric_L2) ? 2 : 1) +
               ((WithNorm && Metric == VecSimMetric_IP) ? 1 : 0);
    }

    // Stored layout: one quantized byte per dimension, followed by metric-specific FP32 metadata.
    template <VecSimMetric Metric, bool WithNorm = false>
    static constexpr size_t storage_bytes_count(size_t dim) {
        return data_bytes_count(dim) + storage_metadata_count<Metric, WithNorm>() * sizeof(float);
    }

    // Index of x_mean_ip / y_mean_ip in the last slot in metadata array
    template <VecSimMetric Metric>
    static constexpr size_t mean_ip_index() {
        return storage_metadata_count<Metric, true>() - 1;
    }

    template <VecSimMetric Metric>
    static constexpr size_t query_mean_ip_index() {
        return query_metadata_count<Metric, true>() - 1;
    }
};

} // namespace vecsim_types
