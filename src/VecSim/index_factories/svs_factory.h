#pragma once

#include <cstdlib> // size_t

#include "VecSim/vec_sim.h"        //typedef VecSimIndex
#include "VecSim/vec_sim_common.h" // VecSimParams, SVSParams

namespace SVSFactory {
VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized = false);
size_t EstimateInitialSize(const SVSParams *params, bool is_normalized = false);
size_t EstimateElementSize(const SVSParams *params);
}; // namespace SVSFactory
