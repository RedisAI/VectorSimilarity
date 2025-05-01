/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/index_factories/brute_force_factory.h"

#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace TieredFactory {

namespace TieredHNSWFactory {

static inline BFParams NewBFParams(const TieredIndexParams *params) {
    auto hnsw_params = params->primaryIndexParams->algoParams.hnswParams;
    BFParams bf_params = {.type = hnsw_params.type,
                          .dim = hnsw_params.dim,
                          .metric = hnsw_params.metric,
                          .multi = hnsw_params.multi,
                          .blockSize = hnsw_params.blockSize};

    return bf_params;
}

template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex(const TieredIndexParams *params) {

    // initialize hnsw index
    // Normalization is done by the frontend index.
    auto *hnsw_index = reinterpret_cast<HNSWIndex<DataType, DistType> *>(
        HNSWFactory::NewIndex(params->primaryIndexParams, true));
    // initialize brute force index

    BFParams bf_params = NewBFParams(params);

    std::shared_ptr<VecSimAllocator> flat_allocator = VecSimAllocator::newVecsimAllocator();
    size_t dataSize = VecSimParams_GetDataSize(bf_params.type, bf_params.dim, bf_params.metric);

    AbstractIndexInitParams abstractInitParams = {.allocator = flat_allocator,
                                                  .dim = bf_params.dim,
                                                  .vecType = bf_params.type,
                                                  .dataSize = dataSize,
                                                  .metric = bf_params.metric,
                                                  .blockSize = bf_params.blockSize,
                                                  .multi = bf_params.multi,
                                                  .logCtx = params->primaryIndexParams->logCtx};
    auto frontendIndex = static_cast<BruteForceIndex<DataType, DistType> *>(
        BruteForceFactory::NewIndex(&bf_params, abstractInitParams, false));

    // Create new tiered hnsw index
    std::shared_ptr<VecSimAllocator> management_layer_allocator =
        VecSimAllocator::newVecsimAllocator();

    return new (management_layer_allocator) TieredHNSWIndex<DataType, DistType>(
        hnsw_index, frontendIndex, *params, management_layer_allocator);
}

inline size_t EstimateInitialSize(const TieredIndexParams *params) {
    HNSWParams hnsw_params = params->primaryIndexParams->algoParams.hnswParams;

    // Add size estimation of VecSimTieredIndex sub indexes.
    // Normalization is done by the frontend index.
    size_t est = HNSWFactory::EstimateInitialSize(&hnsw_params, true);

    // Management layer allocator overhead.
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    est += sizeof(VecSimAllocator) + allocations_overhead;

    // Size of the TieredHNSWIndex struct.
    if (hnsw_params.type == VecSimType_FLOAT32) {
        est += sizeof(TieredHNSWIndex<float, float>);
    } else if (hnsw_params.type == VecSimType_FLOAT64) {
        est += sizeof(TieredHNSWIndex<double, double>);
    } else if (hnsw_params.type == VecSimType_BFLOAT16) {
        est += sizeof(TieredHNSWIndex<bfloat16, float>);
    } else if (hnsw_params.type == VecSimType_FLOAT16) {
        est += sizeof(TieredHNSWIndex<float16, float>);
    } else if (hnsw_params.type == VecSimType_INT8) {
        est += sizeof(TieredHNSWIndex<int8_t, float>);
    } else if (hnsw_params.type == VecSimType_UINT8) {
        est += sizeof(TieredHNSWIndex<uint8_t, float>);
    } else {
        throw std::invalid_argument("Invalid hnsw_params.type");
    }

    return est;
}

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    // Tiered index that contains HNSW index as primary index
    VecSimType type = params->primaryIndexParams->algoParams.hnswParams.type;
    if (type == VecSimType_FLOAT32) {
        return TieredHNSWFactory::NewIndex<float>(params);
    } else if (type == VecSimType_FLOAT64) {
        return TieredHNSWFactory::NewIndex<double>(params);
    } else if (type == VecSimType_BFLOAT16) {
        return TieredHNSWFactory::NewIndex<bfloat16, float>(params);
    } else if (type == VecSimType_FLOAT16) {
        return TieredHNSWFactory::NewIndex<float16, float>(params);
    } else if (type == VecSimType_INT8) {
        return TieredHNSWFactory::NewIndex<int8_t, float>(params);
    } else if (type == VecSimType_UINT8) {
        return TieredHNSWFactory::NewIndex<uint8_t, float>(params);
    }
    return nullptr; // Invalid type.
}
} // namespace TieredHNSWFactory

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    // Tiered index that contains HNSW index as primary index
    if (params->primaryIndexParams->algo == VecSimAlgo_HNSWLIB) {
        return TieredHNSWFactory::NewIndex(params);
    }
    return nullptr; // Invalid algorithm or type.
}
size_t EstimateInitialSize(const TieredIndexParams *params) {

    size_t est = 0;

    BFParams bf_params{};
    if (params->primaryIndexParams->algo == VecSimAlgo_HNSWLIB) {
        est += TieredHNSWFactory::EstimateInitialSize(params);
        bf_params = TieredHNSWFactory::NewBFParams(params);
    }

    est += BruteForceFactory::EstimateInitialSize(&bf_params, false);
    return est;
}

size_t EstimateElementSize(const TieredIndexParams *params) {
    size_t est = 0;
    if (params->primaryIndexParams->algo == VecSimAlgo_HNSWLIB) {
        est = HNSWFactory::EstimateElementSize(&params->primaryIndexParams->algoParams.hnswParams);
    }
    return est;
}

}; // namespace TieredFactory
