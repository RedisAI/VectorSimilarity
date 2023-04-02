/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/algorithms/hnsw/tiered_factory.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "VecSim/algorithms/brute_force/brute_force_factory.h"

#include "hnsw_tiered.h"

namespace TieredFactory {

namespace TieredHNSWFactory {
template <typename DataType, typename DistType = DataType>
inline VecSimIndex *NewIndex(const TieredIndexParams *params,
                             std::shared_ptr<VecSimAllocator> allocator) {
    // Extract hnsw index params
    HNSWParams *hnsw_params = &params->primaryIndexParams->hnswParams;
    // initialize hnsw index
    auto *hnsw_index = reinterpret_cast<HNSWIndex<DataType, DistType> *>(
        HNSWFactory::NewIndex(hnsw_params, allocator));

    // Create new tieredhnsw index
    return new (allocator) TieredHNSWIndex<DataType, DistType>(hnsw_index, *params);
}

inline size_t EstimateInitialSize(const TieredIndexParams *params, BFParams &bf_params_output) {
    HNSWParams hnsw_params = params->primaryIndexParams->hnswParams;

    // Add size estimation of VecSimTieredIndex sub indexes.
    size_t est = HNSWFactory::EstimateInitialSize(&hnsw_params);

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
} // namespace TieredHNSWFactory

VecSimIndex *NewIndex(const TieredIndexParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    // Tiered index that contains HNSW index as primary index
    if (params->primaryIndexParams->algo == VecSimAlgo_HNSWLIB) {
        VecSimType type = params->primaryIndexParams->hnswParams.type;
        if (type == VecSimType_FLOAT32) {
            return TieredHNSWFactory::NewIndex<float>(params, allocator);
        } else if (type == VecSimType_FLOAT64) {
            return TieredHNSWFactory::NewIndex<double>(params, allocator);
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
        est = HNSWFactory::EstimateElementSize(&params->primaryIndexParams->hnswParams);
    }
    return est;
}

}; // namespace TieredFactory