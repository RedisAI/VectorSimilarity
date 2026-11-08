/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3.
 */
#pragma once
#include "VecSim/quantization/turboquant_components.h"

namespace vecsim_types {
template <unsigned Bits>
struct tq {
    static_assert(Bits == 2 || Bits == 3);
    template <typename DataType, VecSimMetric Metric, bool WithMean>
    using Context =
        vecsim::quantization::TurboQuantStorageContext<Bits, DataType, Metric, WithMean>;
    template <VecSimMetric Metric, bool WithMean = false>
    static constexpr size_t storage_bytes_count(size_t dim) {
        static_assert(Metric == VecSimMetric_IP);
        return (dim * Bits + 7) / 8 + 2 * sizeof(float);
    }
};
using tq2 = tq<2>;
using tq3 = tq<3>;
} // namespace vecsim_types
