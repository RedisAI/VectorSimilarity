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
#if HAVE_SVS
#include "VecSim/index_factories/svs_factory.h"
#include "VecSim/algorithms/svs/svs_tiered.h"
#endif

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

#if HAVE_SVS
namespace TieredSVSFactory {

static inline BFParams NewBFParams(const TieredIndexParams *params) {
    auto &svs_params = params->primaryIndexParams->algoParams.svsParams;
    return BFParams{.type = svs_params.type,
                    .dim = svs_params.dim,
                    .metric = svs_params.metric,
                    .multi = false, // multi not supported
                    .blockSize = svs_params.blockSize};
}

template <typename DataType>
inline VecSimIndex *NewIndex(const TieredIndexParams *params) {

    // initialize svs index
    // Normalization is done by the frontend index.
    auto *svs_index = static_cast<VecSimIndexAbstract<DataType, float> *>(
        SVSFactory::NewIndex(params->primaryIndexParams, true));
    assert(svs_index != nullptr);
    // initialize brute force index

    auto bf_params = NewBFParams(params);

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
    auto frontendIndex = static_cast<BruteForceIndex_Single<DataType, float> *>(
        BruteForceFactory::NewIndex(&bf_params, abstractInitParams, false));

    // Create new tiered svs index
    std::shared_ptr<VecSimAllocator> management_layer_allocator =
        VecSimAllocator::newVecsimAllocator();

    return new (management_layer_allocator)
        TieredSVSIndex<DataType>(svs_index, frontendIndex, *params, management_layer_allocator);
}

inline size_t EstimateInitialSize(const TieredIndexParams *params) {
    auto &svs_params = params->primaryIndexParams->algoParams.svsParams;

    // Add size estimation of VecSimTieredIndex sub indexes.
    size_t est = SVSFactory::EstimateInitialSize(&svs_params, true);

    // Management layer allocator overhead.
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    est += sizeof(VecSimAllocator) + allocations_overhead;

    // Size of the TieredHNSWIndex struct.
    switch (svs_params.type) {
    case VecSimType_FLOAT32:
        est += sizeof(TieredSVSIndex<float>);
        break;
    case VecSimType_FLOAT16:
        est += sizeof(TieredSVSIndex<float16>);
        break;
    default:
        assert(false && "Unsupported data type");
        break;
    }

    return est;
}

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    // Tiered index that contains HNSW index as primary index
    VecSimType type = params->primaryIndexParams->algoParams.svsParams.type;
    switch (type) {
    case VecSimType_FLOAT32:
        return TieredSVSFactory::NewIndex<float>(params);
    case VecSimType_FLOAT16:
        return TieredSVSFactory::NewIndex<float16>(params);
    default:
        assert(false && "Unsupported data type");
        return nullptr; // Invalid type.
    }
    return nullptr; // Invalid type.
}
} // namespace TieredSVSFactory
#endif

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    switch (params->primaryIndexParams->algo) {
    // Tiered index that contains HNSW index as primary index
    case VecSimAlgo_HNSWLIB:
        return TieredHNSWFactory::NewIndex(params);
#if HAVE_SVS
        // Tiered index that contains SVS index as primary index
    case VecSimAlgo_SVS:
        return TieredSVSFactory::NewIndex(params);
#endif
        default:
        return nullptr; // Invalid algorithm.
    }
}

size_t EstimateInitialSize(const TieredIndexParams *params) {

    size_t est = 0;

    BFParams bf_params{};
    switch (params->primaryIndexParams->algo) {
    case VecSimAlgo_HNSWLIB:
        est += TieredHNSWFactory::EstimateInitialSize(params);
        bf_params = TieredHNSWFactory::NewBFParams(params);
        break;
#if HAVE_SVS
    case VecSimAlgo_SVS:
        est += TieredSVSFactory::EstimateInitialSize(params);
        bf_params = TieredSVSFactory::NewBFParams(params);
        break;
#endif
    default:
        assert(false && "Invalid algorithm");
    }

    est += BruteForceFactory::EstimateInitialSize(&bf_params, false);
    return est;
}

size_t EstimateElementSize(const TieredIndexParams *params) {
    size_t est = 0;
    switch (params->primaryIndexParams->algo) {
    case VecSimAlgo_HNSWLIB:
        est = HNSWFactory::EstimateElementSize(&params->primaryIndexParams->algoParams.hnswParams);
        break;
#if HAVE_SVS
    case VecSimAlgo_SVS:
        est = SVSFactory::EstimateElementSize(&params->primaryIndexParams->algoParams.svsParams);
        break;
#endif
    default:
        assert(false && "Invalid algorithm");
    }
    return est;
}

}; // namespace TieredFactory
