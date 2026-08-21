/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "NEON_HP.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/L2/L2_NEON_FP16.h"
#include "VecSim/spaces/IP/IP_NEON_FP16.h"
#include "VecSim/spaces/IP/IP_NEON_SQ8_FP16.h"
#include "VecSim/spaces/L2/L2_NEON_SQ8_FP16.h"

namespace spaces {

dist_func_t<float> Choose_FP16_L2_implementation_NEON_HP(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP16_L2Sqr_NEON_HP<N>; });
}

dist_func_t<float> Choose_FP16_IP_implementation_NEON_HP(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP16_InnerProduct_NEON_HP<N>; });
}

dist_func_t<float> Choose_SQ8_FP16_IP_implementation_NEON_HP(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_InnerProductSIMD16_NEON_HP<N>; });
}

dist_func_t<float> Choose_SQ8_FP16_L2_implementation_NEON_HP(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_L2SqrSIMD16_NEON_HP<N>; });
}

dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_NEON_HP(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_CosineSIMD16_NEON_HP<N>; });
}

// FMLAL (FEAT_FHM / asimdfhm) variants.
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_NEON_FHM(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_InnerProductSIMD16_NEON_FHM<N>; });
}

dist_func_t<float> Choose_SQ8_FP16_L2_implementation_NEON_FHM(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_L2SqrSIMD16_NEON_FHM<N>; });
}

dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_NEON_FHM(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_CosineSIMD16_NEON_FHM<N>; });
}

} // namespace spaces
