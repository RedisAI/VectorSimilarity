/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/vec_sim.h"

namespace PiPNNFactory {
// Resolve zero-valued defaults and validate every public setting. Throws invalid_argument or
// overflow_error on invalid input. Shared by construction, estimates, and native loading.
PiPNNParams ResolveParams(const PiPNNParams &params);
VecSimIndex *NewIndex(const VecSimParams *params);
size_t EstimateInitialSize(const PiPNNParams *params);
size_t EstimateElementSize(const PiPNNParams *params);
} // namespace PiPNNFactory
