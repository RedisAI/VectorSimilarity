/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // HNSWParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/vec_sim_index.h"

namespace HNSWFactory {
/** @param is_normalized is used to determine the index's computer type. If the index metric is
 * Cosine, and is_normalized == true, we will create the computer as if the metric is IP, assuming
 * the blobs sent to the index are already normalized. For example, in case it's a tiered index,
 * where the blobs are normalized by the frontend index.
 */
VecSimIndex *NewIndex(const VecSimParams *params, bool is_normalized = false);
VecSimIndex *NewIndex(const HNSWParams *params, bool is_normalized = false);
size_t EstimateInitialSize(const HNSWParams *params, bool is_normalized = false);
size_t EstimateElementSize(const HNSWParams *params);

#ifdef BUILD_TESTS
// Factory function to be used before loading a serialized index.
// @params is only used for backward compatibility with V1. It won't be used if V2 and up is loaded.
// Required fields: type, dim, metric and multi
// Permission fields that *** must be initalized to zero ***: blockSize, epsilon *
VecSimIndex *NewIndex(const std::string &location, bool is_normalized = false);

#endif

}; // namespace HNSWFactory
