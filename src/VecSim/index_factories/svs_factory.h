// TODO: duplicate includes with other headers,
// it can be refactored.
#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/vec_sim_index.h"

namespace SVSFactory {
VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized = false);
size_t EstimateInitialSize(const SVSParams *params, bool is_normalized = false);
size_t EstimateElementSize(const SVSParams *params);
}; // namespace SVSFactory
