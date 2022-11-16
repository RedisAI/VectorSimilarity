/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // HNSWParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator

namespace HNSWFactory {

VecSimIndex *NewIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const HNSWParams *params);
size_t EstimateElementSize(const HNSWParams *params);

}; // namespace HNSWFactory
