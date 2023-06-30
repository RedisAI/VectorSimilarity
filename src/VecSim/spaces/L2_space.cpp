/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_SSE.h"

#include "VecSim/spaces/space_chooser.h"

namespace spaces {

dist_func_t<float> L2_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<float> ret_dist_func = FP32_L2Sqr;
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
        CHOOSE_IMPLEMENTATION(dim, 16, FP32_L2SqrSIMD16Ext_AVX512);
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
        CHOOSE_IMPLEMENTATION(dim, 16, FP32_L2SqrSIMD16Ext_AVX);
#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
        CHOOSE_IMPLEMENTATION(dim, 16, FP32_L2SqrSIMD16Ext_SSE);
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> L2_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<double> ret_dist_func = FP64_L2Sqr;
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
        CHOOSE_IMPLEMENTATION(dim, 8, FP64_L2SqrSIMD8Ext_AVX512);
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
        CHOOSE_IMPLEMENTATION(dim, 8, FP64_L2SqrSIMD8Ext_AVX);
#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
        CHOOSE_IMPLEMENTATION(dim, 8, FP64_L2SqrSIMD8Ext_SSE);
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__ */
    return ret_dist_func;
}

} // namespace spaces

#include "VecSim/spaces/space_chooser_cleanup.h"
