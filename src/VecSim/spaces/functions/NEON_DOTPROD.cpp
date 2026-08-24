/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "NEON.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/IP/IP_NEON_DOTPROD_INT8.h"
#include "VecSim/spaces/IP/IP_NEON_DOTPROD_UINT8.h"
#include "VecSim/spaces/IP/IP_NEON_DOTPROD_SQ8_SQ8.h"
#include "VecSim/spaces/L2/L2_NEON_DOTPROD_INT8.h"
#include "VecSim/spaces/L2/L2_NEON_DOTPROD_UINT8.h"
#include "VecSim/spaces/L2/L2_NEON_DOTPROD_SQ8_SQ8.h"

namespace spaces {

dist_func_t<float> Choose_INT8_IP_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return INT8_InnerProductSIMD16_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_UINT8_IP_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return UINT8_InnerProductSIMD16_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_INT8_Cosine_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return INT8_CosineSIMD_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_UINT8_Cosine_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return UINT8_CosineSIMD_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_INT8_L2_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return INT8_L2SqrSIMD16_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_UINT8_L2_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return UINT8_L2SqrSIMD16_NEON_DOTPROD<N>; });
}

// SQ8-to-SQ8 distance functions (both vectors are uint8 quantized with precomputed sum)
dist_func_t<float> Choose_SQ8_SQ8_IP_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return SQ8_SQ8_InnerProductSIMD64_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_SQ8_SQ8_Cosine_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return SQ8_SQ8_CosineSIMD64_NEON_DOTPROD<N>; });
}

dist_func_t<float> Choose_SQ8_SQ8_L2_implementation_NEON_DOTPROD(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return SQ8_SQ8_L2SqrSIMD64_NEON_DOTPROD<N>; });
}

} // namespace spaces
