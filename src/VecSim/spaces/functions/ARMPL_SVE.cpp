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
    return FP32_InnerProduct_ARMPL_SVE;
}

dist_func_t<float> Choose_FP32_L2_implementation_ARMPL_SVE(size_t dim) {
    return FP32_L2Sqr_ARMPL_SVE;
}

dist_func_t<double> Choose_FP64_IP_implementation_ARMPL_SVE(size_t dim) {
    return FP64_InnerProduct_ARMPL_SVE;
}

dist_func_t<double> Choose_FP64_L2_implementation_ARMPL_SVE(size_t dim) {
    return FP64_L2Sqr_ARMPL_SVE;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
