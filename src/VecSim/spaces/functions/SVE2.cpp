/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "SVE2.h"
#include "VecSim/spaces/L2/L2_SVE2_FP32.h"
#include "VecSim/spaces/IP/IP_SVE2_FP32.h"
#include "VecSim/spaces/L2/L2_SVE2_INT8.h"
#include "VecSim/spaces/IP/IP_SVE_INT8.h" // SVE2 implementation is identical to SVE
#include "VecSim/spaces/L2/L2_SVE2_UINT8.h"
#include "VecSim/spaces/IP/IP_SVE_UINT8.h" // SVE2 implementation is identical to SVE

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP32_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP32_InnerProductSIMD_SVE2, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP32_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, FP32_L2SqrSIMD_SVE2, dim, svcntw);
    return ret_dist_func;
}

dist_func_t<float> Choose_INT8_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, INT8_L2SqrSIMD_SVE2, dim, svcntb);
    return ret_dist_func;
}

dist_func_t<float> Choose_INT8_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, INT8_InnerProductSIMD_SVE, dim, svcntb);
    return ret_dist_func;
}

dist_func_t<float> Choose_INT8_Cosine_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, INT8_CosineSIMD_SVE, dim, svcntb);
    return ret_dist_func;
}

dist_func_t<float> Choose_UINT8_L2_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, UINT8_L2SqrSIMD_SVE2, dim, svcntb);
    return ret_dist_func;
}

dist_func_t<float> Choose_UINT8_IP_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, UINT8_InnerProductSIMD_SVE, dim, svcntb);
    return ret_dist_func;
}

dist_func_t<float> Choose_UINT8_Cosine_implementation_SVE2(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_dist_func, UINT8_CosineSIMD_SVE, dim, svcntb);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
