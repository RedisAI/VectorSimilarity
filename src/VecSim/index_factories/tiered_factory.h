/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once

#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/index_factories/components/preprocessors_factory.h"

namespace TieredFactory {

VecSimIndex *NewIndex(const TieredIndexParams *params);

// The size estimation is the sum of the buffer (brute force) and main index initial sizes
// estimations, plus the tiered index class size. Note it does not include the size of internal
// containers such as the job queue, as those depend on the user implementation.
size_t EstimateInitialSize(const TieredIndexParams *params);
size_t EstimateElementSize(const TieredIndexParams *params);

#ifdef BUILD_TESTS
namespace TieredHNSWFactory {
// Build tiered index from existing HNSW index - for internal benchmarks purposes
template <typename DataType, typename DistType>
VecSimIndex *NewIndex(const TieredIndexParams *params, HNSWIndex<DataType, DistType> *hnsw_index) {
    // Initialize brute force index.
    BFParams bf_params = {.type = hnsw_index->getType(),
                          .dim = hnsw_index->getDim(),
                          .metric = hnsw_index->getMetric(),
                          .multi = hnsw_index->isMultiValue(),
                          .blockSize = hnsw_index->getBlockSize()};

    std::shared_ptr<VecSimAllocator> flat_allocator = VecSimAllocator::newVecsimAllocator();
    AbstractIndexInitParams abstractInitParams = {.allocator = flat_allocator,
                                                  .dim = bf_params.dim,
                                                  .vecType = bf_params.type,
                                                  .metric = bf_params.metric,
                                                  .blockSize = bf_params.blockSize,
                                                  .multi = bf_params.multi,
                                                  .logCtx = nullptr};
    auto frontendIndex = static_cast<BruteForceIndex<DataType, DistType> *>(
        BruteForceFactory::NewIndex(&bf_params, abstractInitParams, true));

    // Create new tiered hnsw index
    std::shared_ptr<VecSimAllocator> management_layer_allocator =
        VecSimAllocator::newVecsimAllocator();

    // Create preprocessors container
    auto preprocessors = CreatePreprocessorsContainer<DataType>(
        management_layer_allocator, {.metric = bf_params.metric,
                                     .dim = bf_params.dim,
                                     .alignment = frontendIndex->getAlignment()});

    return new (management_layer_allocator) TieredHNSWIndex<DataType, DistType>(
        hnsw_index, frontendIndex, *params, management_layer_allocator, preprocessors);
}
} // namespace TieredHNSWFactory
#endif

}; // namespace TieredFactory
