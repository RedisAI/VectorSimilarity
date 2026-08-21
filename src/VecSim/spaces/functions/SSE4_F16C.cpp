/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "SSE4_F16C.h"
#include "VecSim/spaces/functions/residual_dispatch.h"
#include "VecSim/spaces/IP/IP_SSE4_SQ8_FP16.h"
#include "VecSim/spaces/L2/L2_SSE4_SQ8_FP16.h"

namespace spaces {

dist_func_t<float> Choose_SQ8_FP16_IP_implementation_SSE4(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_InnerProductSIMD16_SSE4<N>; });
}
dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_SSE4(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_CosineSIMD16_SSE4<N>; });
}
dist_func_t<float> Choose_SQ8_FP16_L2_implementation_SSE4(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_L2SqrSIMD16_SSE4<N>; });
}

} // namespace spaces
