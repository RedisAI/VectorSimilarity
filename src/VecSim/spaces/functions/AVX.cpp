/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "AVX.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/L2/L2_AVX_FP32.h"
#include "VecSim/spaces/L2/L2_AVX_FP64.h"

#include "VecSim/spaces/IP/IP_AVX_FP32.h"
#include "VecSim/spaces/IP/IP_AVX_FP64.h"

namespace spaces {

dist_func_t<float> Choose_FP32_IP_implementation_AVX(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP32_InnerProductSIMD16_AVX<N>; });
}

dist_func_t<double> Choose_FP64_IP_implementation_AVX(size_t dim) {
    return dispatch_by_residual<dist_func_t<double>, 8>(
        dim, []<size_t N>() { return FP64_InnerProductSIMD8_AVX<N>; });
}

dist_func_t<float> Choose_FP32_L2_implementation_AVX(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP32_L2SqrSIMD16_AVX<N>; });
}

dist_func_t<double> Choose_FP64_L2_implementation_AVX(size_t dim) {
    return dispatch_by_residual<dist_func_t<double>, 8>(
        dim, []<size_t N>() { return FP64_L2SqrSIMD8_AVX<N>; });
}

} // namespace spaces
