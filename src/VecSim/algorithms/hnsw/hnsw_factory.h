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
VecSimIndex *NewTieredIndex(const TieredHNSWParams *params,
                            std::shared_ptr<VecSimAllocator> allocator);
size_t EstimateInitialSize(const HNSWParams *params);
size_t EstimateElementSize(const HNSWParams *params);

#ifdef BUILD_TESTS
// Factory function to be used before loading a serialized index.
// @params is only used for backward compatibility with V1. It won't be used if V2 and up is loaded.
// Required fields: type, dim, metric and multi
// Permission fields that *** must be initalized to zero ***: blockSize, epsilon *
VecSimIndex *NewIndex(const std::string &location, const HNSWParams *v1_params = nullptr);

#endif
}; // namespace HNSWFactory
