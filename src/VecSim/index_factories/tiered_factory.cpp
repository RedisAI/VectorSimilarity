/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/index_factories/brute_force_factory.h"

#include "VecSim/algorithms/hnsw/hnsw_tiered.h"

namespace TieredFactory {

namespace TieredHNSWFactory {
template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex(const TieredIndexParams *params) {

    // initialize hnsw index
    auto *hnsw_index = reinterpret_cast<HNSWIndex<DataType, DistType> *>(
        HNSWFactory::NewIndex(params->primaryIndexParams));
    // initialize brute force index

    BFParams bf_params = {.type = params->primaryIndexParams->algoParams.hnswParams.type,
                          .dim = params->primaryIndexParams->algoParams.hnswParams.dim,
                          .metric = params->primaryIndexParams->algoParams.hnswParams.metric,
                          .multi = params->primaryIndexParams->algoParams.hnswParams.multi,
                          .blockSize = params->primaryIndexParams->algoParams.hnswParams.blockSize};

    std::shared_ptr<VecSimAllocator> flat_allocator = VecSimAllocator::newVecsimAllocator();
    AbstractIndexInitParams abstractInitParams = {.allocator = flat_allocator,
                                                  .dim = bf_params.dim,
                                                  .vecType = bf_params.type,
                                                  .metric = bf_params.metric,
                                                  .blockSize = bf_params.blockSize,
                                                  .multi = bf_params.multi,
                                                  .logCtx = params->primaryIndexParams->logCtx};
    auto frontendIndex = static_cast<BruteForceIndex<DataType, DistType> *>(
        BruteForceFactory::NewIndex(&bf_params, abstractInitParams));

    // Create new tiered hnsw index
    std::shared_ptr<VecSimAllocator> management_layer_allocator =
        VecSimAllocator::newVecsimAllocator();

    return new (management_layer_allocator) TieredHNSWIndex<DataType, DistType>(
        hnsw_index, frontendIndex, *params, management_layer_allocator);
}

inline size_t EstimateInitialSize(const TieredIndexParams *params, BFParams &bf_params_output) {
    HNSWParams hnsw_params = params->primaryIndexParams->algoParams.hnswParams;

    // Add size estimation of VecSimTieredIndex sub indexes.
    size_t est = HNSWFactory::EstimateInitialSize(&hnsw_params);

    // Management layer allocator overhead.
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    est += sizeof(VecSimAllocator) + allocations_overhead;

    // Size of the TieredHNSWIndex struct.
    if (hnsw_params.type == VecSimType_FLOAT32) {
        est += sizeof(TieredHNSWIndex<float, float>);
    } else if (hnsw_params.type == VecSimType_FLOAT64) {
        est += sizeof(TieredHNSWIndex<double, double>);
    }
    bf_params_output.type = hnsw_params.type;
    bf_params_output.multi = hnsw_params.multi;

    return est;
}

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    // Tiered index that contains HNSW index as primary index
    VecSimType type = params->primaryIndexParams->algoParams.hnswParams.type;
    if (type == VecSimType_FLOAT32) {
        return TieredHNSWFactory::NewIndex<float>(params);
    } else if (type == VecSimType_FLOAT64) {
        return TieredHNSWFactory::NewIndex<double>(params);
    }
    return nullptr; // Invalid type.
}
} // namespace TieredHNSWFactory

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    // Tiered index that contains HNSW index as primary index
    if (params->primaryIndexParams->algo == VecSimAlgo_HNSWLIB) {
        VecSimType type = params->primaryIndexParams->algoParams.hnswParams.type;
        if (type == VecSimType_FLOAT32) {
            return TieredHNSWFactory::NewIndex<float>(params);
        } else if (type == VecSimType_FLOAT64) {
            return TieredHNSWFactory::NewIndex<double>(params);
        }
    }
    return nullptr; // Invalid algorithm or type.
}
size_t EstimateInitialSize(const TieredIndexParams *params) {

    size_t est = 0;

    BFParams bf_params{};
    if (params->primaryIndexParams->algo == VecSimAlgo_HNSWLIB) {
        est += TieredHNSWFactory::EstimateInitialSize(params, bf_params);
    }

    est += BruteForceFactory::EstimateInitialSize(&bf_params);
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
