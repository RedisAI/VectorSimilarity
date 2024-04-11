/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "AVX512.h"

#include "VecSim/spaces/L2/L2_AVX512_FP32.h"
#include "VecSim/spaces/L2/L2_AVX512_FP64.h"

#include "VecSim/spaces/IP/IP_AVX512_FP32.h"
#include "VecSim/spaces/IP/IP_AVX512_FP64.h"
#include "VecSim/spaces/IP/IP_AVX512_FP16.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP32_IP_implementation_AVX512(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_InnerProductSIMD16_AVX512);
    return ret_dist_func;
}

dist_func_t<double> Choose_FP64_IP_implementation_AVX512(size_t dim) {
    dist_func_t<double> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_InnerProductSIMD8_AVX512);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP16_IP_implementation_AVX512(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP16_InnerProductSIMD16_AVX512);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP32_L2_implementation_AVX512(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2SqrSIMD16_AVX512);
    return ret_dist_func;
}

dist_func_t<double> Choose_FP64_L2_implementation_AVX512(size_t dim) {
    dist_func_t<double> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_L2SqrSIMD8_AVX512);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
