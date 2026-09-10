/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/L2/L2.h"

template <bool partial_chunk, unsigned char additional_steps>
float SQ8_FP32_L2SqrSIMD_SVE(const void *storage, const void *query, size_t dimension) {
    return SQ8_FP32_L2Sqr(storage, query, dimension);
}
