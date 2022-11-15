#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // NGTParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

namespace NGTFactory {

VecSimIndex *NewIndex(const NGTParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const NGTParams *params);
size_t EstimateElementSize(const NGTParams *params);

}; // namespace NGTFactory
