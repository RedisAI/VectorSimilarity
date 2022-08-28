#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
template <typename DataType, typename DistType>
class BruteForceIndex;

class BruteForceFactory {

public:
    static VecSimIndex *NewIndex(const BFParams *params,
                                 std::shared_ptr<VecSimAllocator> allocator);
    static size_t EstimateInitialSize(const BFParams *params);
    static size_t EstimateElementSize(const BFParams *params);
    static VecSimBatchIterator *newBatchIterator(void *queryBlob, VecSimQueryParams *queryParams,
                                                 std::shared_ptr<VecSimAllocator> allocator,
                                                 const BruteForceIndex<float, float> *index);
};
