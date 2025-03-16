/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "ARMPL_NEON.h"
#include "VecSim/spaces/L2/L2_ARMPL_NEON_FP32.h"
#include "VecSim/spaces/IP/IP_ARMPL_NEON_FP32.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP32_IP_implementation_ARMPL_NEON(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_InnerProductSIMD16_ARMPL_NEON);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP32_L2_implementation_ARMPL_NEON(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2SqrSIMD16_ARMPL_NEON);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
