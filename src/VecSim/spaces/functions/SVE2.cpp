/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "NEON.h"
#include "VecSim/spaces/L2/L2_SVE2_FP32.h"
#include "VecSim/spaces/IP/IP_SVE2_FP32.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP32_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    switch (svcntw()) {
        case 8:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP32_InnerProductSIMD_SVE2);
            break;
        case 16:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_InnerProductSIMD_SVE2);
            break;        
        case 32:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP32_InnerProductSIMD_SVE2);
            break;        
        case 64:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 64, FP32_InnerProductSIMD_SVE2);
            break;
        default:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 4, FP32_InnerProductSIMD_SVE2);
    }
    return ret_dist_func;
}

dist_func_t<float> Choose_FP32_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    switch (svcntw()) {
        case 8:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP32_L2SqrSIMD_SVE2);
            break;
        case 16:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2SqrSIMD_SVE2);
            break;        
        case 32:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP32_L2SqrSIMD_SVE2);
            break;        
        case 64:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 64, FP32_L2SqrSIMD_SVE2);
            break;
        default:
            CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 4, FP32_L2SqrSIMD_SVE2);
    }
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
