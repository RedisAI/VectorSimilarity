/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "SVE2.h"

#include "VecSim/spaces/L2/L2_SVE_FP32.h"
#include "VecSim/spaces/IP/IP_SVE_FP32.h"

#include "VecSim/spaces/IP/IP_SVE_FP16.h" // using SVE implementation, but different compilation flags
#include "VecSim/spaces/L2/L2_SVE_FP16.h" // using SVE implementation, but different compilation flags

#include "VecSim/spaces/IP/IP_SVE_FP64.h"
#include "VecSim/spaces/L2/L2_SVE_FP64.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP32_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP32_InnerProductSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}
dist_func_t<float> Choose_FP32_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP32_L2SqrSIMD_SVE, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP16_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP16_InnerProduct_SVE, dim, svcnth);
    return ret_dist_func;
}
dist_func_t<float> Choose_FP16_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP16_L2Sqr_SVE, dim, svcnth);
    return ret_dist_func;
}

dist_func_t<double> Choose_FP64_IP_implementation_SVE2(size_t dim) {
    dist_func_t<double> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP64_InnerProductSIMD_SVE, dim, svcntd);
    return ret_dist_func;
}
dist_func_t<double> Choose_FP64_L2_implementation_SVE2(size_t dim) {
    dist_func_t<double> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP64_L2SqrSIMD_SVE, dim, svcntd);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
