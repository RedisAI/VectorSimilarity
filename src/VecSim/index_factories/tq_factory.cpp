/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/index_factories/tq_factory.h"

#include "VecSim/algorithms/tq/tq_flat.h"

namespace TQFactory {
namespace {

AbstractIndexInitParams NewAbstractInitParams(const TQFlatParams *params, void *logCtx,
                                              std::shared_ptr<VecSimAllocator> allocator,
                                              size_t stored_data_size) {
    return {.allocator = allocator,
            .dim = params->dim,
            .vecType = params->type,
            .storedDataSize = stored_data_size,
            .metric = params->metric,
            .blockSize = params->blockSize,
            .multi = false,
            .isDisk = false,
            .logCtx = logCtx,
            .inputBlobSize = params->dim * VecSimType_sizeof(params->type)};
}

template <VecSimMetric Metric>
VecSimIndex *NewIndexImpl(const VecSimParams *params) {
    const auto &tq_params = params->algoParams.tqFlatParams;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto components = TQFlatDetails::CreateTQComponents<Metric>(allocator, &tq_params);
    auto stored_data_size = TQFlatDetails::GetStorageDataSize<Metric>(&tq_params);
    auto abstract_init_params =
        NewAbstractInitParams(&tq_params, params->logCtx, allocator, stored_data_size);
    BFParams bf_params = {.type = tq_params.type,
                          .dim = tq_params.dim,
                          .metric = tq_params.metric,
                          .multi = false,
                          .initialCapacity = tq_params.initialCapacity,
                          .blockSize = tq_params.blockSize};
    return new (allocator)
        TQFlatDetails::TQFlatIndex(&bf_params, abstract_init_params, components);
}

template <VecSimMetric Metric>
size_t EstimateInitialSizeImpl(const TQFlatParams *params) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;
    est += sizeof(TQFlatDetails::TQFlatIndex);
    est += sizeof(DataBlocksContainer) + allocations_overhead;
    est += allocations_overhead + sizeof(TQFlatDetails::TQDistanceCalculator<Metric>);
    est += allocations_overhead + sizeof(MultiPreprocessorsContainer<float, 1>);
    est += allocations_overhead + sizeof(TQFlatDetails::TQPreprocessor<Metric>);
    est += params->dim * params->dim * sizeof(float);
    est += params->projections * params->dim * sizeof(float);
    est += (size_t{1} << (params->bits - 1)) * 2 * sizeof(float);
    return est;
}

template <VecSimMetric Metric>
size_t EstimateElementSizeImpl(const TQFlatParams *params) {
    return TQFlatDetails::GetStorageDataSize<Metric>(params) + sizeof(labelType) +
           sizeof(void *);
}

} // namespace

VecSimIndex *NewIndex(const VecSimParams *params) {
    const auto &tq_params = params->algoParams.tqFlatParams;
    if (tq_params.type != VecSimType_FLOAT32) {
        throw std::invalid_argument("TQ-FLAT currently supports FLOAT32 input only");
    }
    if (tq_params.multi) {
        throw std::invalid_argument("TQ-FLAT currently supports single-value indexes only");
    }

    switch (tq_params.metric) {
    case VecSimMetric_L2:
        return NewIndexImpl<VecSimMetric_L2>(params);
    case VecSimMetric_IP:
        return NewIndexImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return NewIndexImpl<VecSimMetric_Cosine>(params);
    }
    return nullptr;
}

size_t EstimateInitialSize(const TQFlatParams *params) {
    switch (params->metric) {
    case VecSimMetric_L2:
        return EstimateInitialSizeImpl<VecSimMetric_L2>(params);
    case VecSimMetric_IP:
        return EstimateInitialSizeImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return EstimateInitialSizeImpl<VecSimMetric_Cosine>(params);
    }
    return 0;
}

size_t EstimateElementSize(const TQFlatParams *params) {
    switch (params->metric) {
    case VecSimMetric_L2:
        return EstimateElementSizeImpl<VecSimMetric_L2>(params);
    case VecSimMetric_IP:
        return EstimateElementSizeImpl<VecSimMetric_IP>(params);
    case VecSimMetric_Cosine:
        return EstimateElementSizeImpl<VecSimMetric_Cosine>(params);
    }
    return 0;
}

} // namespace TQFactory
