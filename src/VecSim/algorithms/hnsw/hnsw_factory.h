#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // HNSWParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

template <typename DataType, typename DistType>
class HNSWIndex;

namespace HNSWFactory {

VecSimIndex *NewIndex(const HNSWParams *params,
                                 std::shared_ptr<VecSimAllocator> allocator);
 size_t EstimateInitialSize(const HNSWParams *params);
size_t EstimateElementSize(const HNSWParams *params);


VecSimBatchIterator *newBatchIterator(void *queryBlob, VecSimQueryParams *queryParams,
                                                 std::shared_ptr<VecSimAllocator> allocator,
                                                 HNSWIndex<float, float> *index);    

}; //namespace HNSWFactory
