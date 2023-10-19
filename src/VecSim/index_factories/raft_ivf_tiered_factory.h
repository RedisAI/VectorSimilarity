#pragma once

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // RaftIvfParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/vec_sim_index.h"

namespace TieredRaftIvfFactory {

template <typename DataType, typename DistType = DataType>
VecSimIndex *NewIndex(const TieredIndexParams *params);

// The size estimation is the sum of the buffer (brute force) and main index initial sizes
// estimations, plus the tiered index class size. Note it does not include the size of internal
// containers such as the job queue, as those depend on the user implementation.
size_t EstimateInitialSize(const TieredIndexParams *params);
size_t EstimateElementSize(const TieredIndexParams *params);

}; // namespace TieredRaftIvfFactory
