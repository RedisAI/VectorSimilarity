#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // HNSWParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
class HNSWFactory {

public:
    static VecSimIndex *NewIndex(const HNSWParams *params,
                                 std::shared_ptr<VecSimAllocator> allocator);
    static size_t EstimateInitialSize(const HNSWParams *params);
    static size_t EstimateElementSize(const HNSWParams *params);
    static VecSimBatchIterator *newBatchIterator(void *queryBlob, VecSimQueryParams *queryParams,
                                                 std::shared_ptr<VecSimAllocator> allocator,
                                                 HNSWIndex *index);
};
