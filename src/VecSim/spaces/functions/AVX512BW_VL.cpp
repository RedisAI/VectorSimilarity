/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "AVX512BW_VL.h"

#include "VecSim/spaces/IP/IP_AVX512BW_VL_FP16.h"
#include "VecSim/spaces/L2/L2_AVX512BW_VL_FP16.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP16_IP_implementation_AVX512BW_VL(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP16_InnerProductSIMD32_AVX512BW_VL);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP16_L2_implementation_AVX512BW_VL(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP16_L2SqrSIMD32_AVX512BW_VL);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
