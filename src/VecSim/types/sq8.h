/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include "VecSim/vec_sim_common.h"

namespace vecsim_types {

// Represents a scalar-quantized 8-bit blob with reconstruction metadata
struct sq8 {
    using value_type = uint8_t;

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
    // WithNorm: one extra metadata slot for x_mean_ip / y_mean_ip
    template <VecSimMetric Metric, bool WithNorm = false>
    static constexpr size_t storage_metadata_count() {
        return ((Metric == VecSimMetric_L2) ? 4 : 3) + (WithNorm ? 1 : 0);
    }

    template <VecSimMetric Metric, bool WithNorm = false>
    static constexpr size_t query_metadata_count() {
        return ((Metric == VecSimMetric_L2) ? 2 : 1) + (WithNorm ? 1 : 0);
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
