/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "NEON_BF16.h"

// Hoisted above the anonymous namespace below so that the standard library and the shared
// type headers keep external linkage. Wrapping them would pull <cstring> and friends into
// the anonymous namespace and fail to compile.
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8.h"
#include <arm_neon.h>

// Kernel instantiations get internal linkage, unique to this translation unit, so two tiers
// that share a kernel header cannot emit the same weak symbol and let link order pick the
// body. Only this tier's Choose_* entry points stay external.
namespace {
#include "VecSim/spaces/L2/L2_NEON_BF16.h"
#include "VecSim/spaces/IP/IP_NEON_BF16.h"
} // namespace

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_BF16_L2_implementation_NEON_BF16(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, BF16_L2Sqr_NEON);
    return ret_dist_func;
}

dist_func_t<float> Choose_BF16_IP_implementation_NEON_BF16(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, BF16_InnerProduct_NEON);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
