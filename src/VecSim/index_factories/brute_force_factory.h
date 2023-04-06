/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // BFParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

namespace BruteForceFactory {
VecSimIndex *NewIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const BFParams *params);
size_t EstimateElementSize(const BFParams *params);
}; // namespace BruteForceFactory
