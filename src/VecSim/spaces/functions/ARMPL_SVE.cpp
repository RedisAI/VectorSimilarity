/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "ARMPL_SVE.h"

#include "VecSim/spaces/L2/L2_ARMPL_SVE_FP32.h"
#include "VecSim/spaces/IP/IP_ARMPL_SVE_FP32.h"
#include "VecSim/spaces/IP/IP_ARMPL_SVE_FP64.h"
#include "VecSim/spaces/L2/L2_ARMPL_SVE_FP64.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP32_IP_implementation_ARMPL_SVE(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_InnerProduct_ARMPL_SVE);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP32_L2_implementation_ARMPL_SVE(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2Sqr_ARMPL_SVE);
    return ret_dist_func;
}

dist_func_t<double> Choose_FP64_IP_implementation_ARMPL_SVE(size_t dim) {
    dist_func_t<double> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_InnerProduct_ARMPL_SVE);
    return ret_dist_func;
}

dist_func_t<double> Choose_FP64_L2_implementation_ARMPL_SVE(size_t dim) {
    dist_func_t<double> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_L2Sqr_ARMPL_SVE);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
