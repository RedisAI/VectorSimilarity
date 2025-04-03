/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/spaces/spaces.h"

namespace spaces {

dist_func_t<float> Choose_BF16_IP_implementation_NEON_BF16(size_t dim);

dist_func_t<float> Choose_BF16_L2_implementation_NEON_BF16(size_t dim);

} // namespace spaces
