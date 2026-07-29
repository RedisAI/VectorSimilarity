/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "AVX512F.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/L2/L2_AVX512F_FP16.h"
#include "VecSim/spaces/L2/L2_AVX512F_FP32.h"
#include "VecSim/spaces/L2/L2_AVX512F_FP64.h"
#include "VecSim/spaces/L2/L2_AVX512F_SQ8_FP16.h"

#include "VecSim/spaces/IP/IP_AVX512F_FP16.h"
#include "VecSim/spaces/IP/IP_AVX512F_FP32.h"
#include "VecSim/spaces/IP/IP_AVX512F_FP64.h"
#include "VecSim/spaces/IP/IP_AVX512F_SQ8_FP16.h"

namespace spaces {

dist_func_t<float> Choose_FP32_IP_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP32_InnerProductSIMD16_AVX512<N>; });
}

dist_func_t<double> Choose_FP64_IP_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<double>, 16>(
        dim, []<size_t N>() { return FP64_InnerProductSIMD8_AVX512<N>; });
}

dist_func_t<float> Choose_FP32_L2_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP32_L2SqrSIMD16_AVX512<N>; });
}

dist_func_t<double> Choose_FP64_L2_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<double>, 16>(
        dim, []<size_t N>() { return FP64_L2SqrSIMD8_AVX512<N>; });
}

dist_func_t<float> Choose_FP16_IP_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP16_InnerProductSIMD32_AVX512<N>; });
}

dist_func_t<float> Choose_FP16_L2_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP16_L2SqrSIMD32_AVX512<N>; });
}

// SQ8↔FP16 kernels only use AVX-512F (cvtph_ps + FMA), so they register here rather than under
// the VNNI tier — CPUs with AVX-512F but no VNNI (Skylake-X, some Cascade Lake variants) can use
// these kernels.
dist_func_t<float> Choose_SQ8_FP16_IP_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_InnerProductSIMD16_AVX512F<N>; });
}
dist_func_t<float> Choose_SQ8_FP16_Cosine_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_CosineSIMD16_AVX512F<N>; });
}
dist_func_t<float> Choose_SQ8_FP16_L2_implementation_AVX512F(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 16>(
        dim, []<size_t N>() { return SQ8_FP16_L2SqrSIMD16_AVX512F<N>; });
}

} // namespace spaces
