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

#include "VecSim/spaces/L2/L2_NEON_FP32.h"
#include "VecSim/spaces/IP/IP_NEON_FP32.h"
#include "VecSim/spaces/L2/L2_NEON_INT8.h"
#include "VecSim/spaces/L2/L2_NEON_UINT8.h"
#include "VecSim/spaces/IP/IP_NEON_INT8.h"
#include "VecSim/spaces/IP/IP_NEON_UINT8.h"
#include "VecSim/spaces/L2/L2_NEON_FP64.h"
#include "VecSim/spaces/IP/IP_NEON_FP64.h"
#include "VecSim/spaces/L2/L2_NEON_SQ8_FP32.h"
#include "VecSim/spaces/IP/IP_NEON_SQ8_FP32.h"
#include "VecSim/spaces/IP/IP_NEON_SQ8_SQ8.h"
#include "VecSim/spaces/L2/L2_NEON_SQ8_SQ8.h"

namespace spaces {

dist_func_t<float> Choose_INT8_IP_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return INT8_InnerProductSIMD16_NEON<N>; });
}

dist_func_t<float> Choose_UINT8_IP_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return UINT8_InnerProductSIMD16_NEON<N>; });
}

dist_func_t<float> Choose_FP32_IP_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return FP32_InnerProductSIMD16_NEON<N>; });
}

dist_func_t<double> Choose_FP64_IP_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<double>, 8>(
        dim, []<size_t N>() { return FP64_InnerProductSIMD8_NEON<N>; });
}

dist_func_t<float> Choose_INT8_Cosine_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return INT8_CosineSIMD_NEON<N>; });
}

dist_func_t<float> Choose_UINT8_Cosine_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return UINT8_CosineSIMD_NEON<N>; });
}

dist_func_t<float> Choose_FP32_L2_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return FP32_L2SqrSIMD16_NEON<N>; });
}
dist_func_t<float> Choose_INT8_L2_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return INT8_L2SqrSIMD16_NEON<N>; });
}

dist_func_t<float> Choose_UINT8_L2_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return UINT8_L2SqrSIMD16_NEON<N>; });
}

dist_func_t<double> Choose_FP64_L2_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<double>, 8>(
        dim, []<size_t N>() { return FP64_L2SqrSIMD8_NEON<N>; });
}

dist_func_t<float> Choose_SQ8_FP32_L2_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP32_L2SqrSIMD16_NEON<N>; });
}

dist_func_t<float> Choose_SQ8_FP32_IP_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP32_InnerProductSIMD16_NEON<N>; });
}

dist_func_t<float> Choose_SQ8_FP32_Cosine_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP32_CosineSIMD16_NEON<N>; });
}

// SQ8-to-SQ8 distance functions (both vectors are uint8 quantized with precomputed sum)
// Uses 64-element chunking to leverage efficient UINT8_InnerProductImp
dist_func_t<float> Choose_SQ8_SQ8_IP_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return SQ8_SQ8_InnerProductSIMD64_NEON<N>; });
}

dist_func_t<float> Choose_SQ8_SQ8_Cosine_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return SQ8_SQ8_CosineSIMD64_NEON<N>; });
}

dist_func_t<float> Choose_SQ8_SQ8_L2_implementation_NEON(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 64>(
        dim, []<size_t N>() { return SQ8_SQ8_L2SqrSIMD64_NEON<N>; });
}

} // namespace spaces
