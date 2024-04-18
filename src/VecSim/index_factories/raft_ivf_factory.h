#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // RaftIvfParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/vec_sim_index.h"

namespace RaftIvfFactory {

VecSimIndex *NewIndex(const VecSimParams *params);
VecSimIndex *NewIndex(const RaftIvfParams *params);
size_t EstimateInitialSize(const RaftIvfParams *params);
size_t EstimateElementSize(const RaftIvfParams *params);

}; // namespace RaftIvfFactory
