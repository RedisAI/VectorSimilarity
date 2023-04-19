#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // RaftIVFFlatParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

namespace RaftIVFFlatFactory {
VecSimIndex *NewIndex(const RaftIVFFlatParams *params, std::shared_ptr<VecSimAllocator> allocator);
VecSimIndex *NewTieredIndex(const TieredRaftIVFFlatParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const RaftIVFFlatParams *params);
size_t EstimateElementSize(const RaftIVFFlatParams *params);
}; // namespace RaftIVFFlatFactory

namespace RaftIVFPQFactory {
VecSimIndex *NewIndex(const RaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
VecSimIndex *NewTieredIndex(const TieredRaftIVFPQParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const RaftIVFPQParams *params);
size_t EstimateElementSize(const RaftIVFPQParams *params);
}; // namespace RaftIVFPQFactory