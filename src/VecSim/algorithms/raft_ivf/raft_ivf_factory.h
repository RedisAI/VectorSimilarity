#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // RaftFlatParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

namespace RaftFlatFactory {
VecSimIndex *NewIndex(const RaftFlatParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const RaftFlatParams *params);
size_t EstimateElementSize(const RaftFlatParams *params);
}; // namespace RaftFlatFactory

namespace RaftPQFactory {
VecSimIndex *NewIndex(const RaftPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const RaftPQParams *params);
size_t EstimateElementSize(const RaftPQParams *params);
}; // namespace RaftPQFactory