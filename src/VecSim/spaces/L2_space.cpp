/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/L2/L2.h"
#if defined(__x86_64__)
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_SSE.h"
#endif

#include "VecSim/spaces/implementation_chooser.h"

namespace spaces {

dist_func_t<float> L2_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<float> ret_dist_func = FP32_L2Sqr;
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2SqrSIMD16Ext_AVX512);
        break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2SqrSIMD16Ext_AVX);
        break;
#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, FP32_L2SqrSIMD16Ext_SSE);
        break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> L2_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<double> ret_dist_func = FP64_L2Sqr;
    // Optimizations assume at least 8 doubles. If we have less, we use the naive implementation.
    if (dim < 8) {
        return ret_dist_func;
    }
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_L2SqrSIMD8Ext_AVX512);
        break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_L2SqrSIMD8Ext_AVX);
        break;
#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
        CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 8, FP64_L2SqrSIMD8Ext_SSE);
        break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__ */
    return ret_dist_func;
}

} // namespace spaces

#include "VecSim/spaces/implementation_chooser_cleanup.h"
