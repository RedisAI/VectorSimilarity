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

    // Template on metric — compile-time constant when metric is known
    template <VecSimMetric Metric>
    static constexpr size_t metadata_count() {
        return (Metric == VecSimMetric_L2) ? 4 : 3;
    }
};

} // namespace vecsim_types
