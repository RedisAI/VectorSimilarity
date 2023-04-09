/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */
#pragma once

#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/memory/vecsim_malloc.h"

namespace TieredFactory {
namespace TieredHNSWFactory {
VecSimIndex *NewIndex(const TieredHNSWParams *params, std::shared_ptr<VecSimAllocator> allocator);
}

VecSimIndex *NewIndex(const TieredIndexParams *params, std::shared_ptr<VecSimAllocator> allocator);

// The size estimation is the sum of the buffer (brute force) and main index initial sizes
// estimations, plus the tiered index class size. Note it does not include the size of internal
// containers such as the job queue, as those depend on the user implementation.
size_t EstimateInitialSize(const TieredIndexParams *params);
size_t EstimateElementSize(const TieredIndexParams *params);

}; // namespace TieredFactory
