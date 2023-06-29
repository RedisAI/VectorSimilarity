/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"

#define CASES16(X, func) C4(X, func, 0) C4(X, func, 1) C4(X, func, 2) C4(X, func, 3)
#define CASES8(X, func) C4(X, func, 0) C4(X, func, 1)
#define C4(X, func, N) X((4*N), func) X((4*N+1), func) X((4*N+2), func) X((4*N+3), func)
#define X(N, func) case (N):            \
  ret_dist_func = func<(1 << (N)) - 1>; \
  break;

namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<float> ret_dist_func = FP32_InnerProduct;
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
    {
        switch (dim % 16) {
            CASES16(X, FP32_InnerProductSIMD16Ext_AVX512);
        }
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
    {
        switch (dim % 16) {
            CASES16(X, FP32_InnerProductSIMD16Ext_AVX);
        }
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        switch (dim % 16) {
            CASES16(X, FP32_InnerProductSIMD16Ext_SSE);
        }
    } break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<double> ret_dist_func = FP64_InnerProduct;
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
    {
        switch (dim % 8) {
            CASES8(X, FP64_InnerProductSIMD8Ext_AVX512);
        }
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
    {
        switch (dim % 8) {
            CASES8(X, FP64_InnerProductSIMD8Ext_AVX);
        }
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        switch (dim % 8) {
            CASES8(X, FP64_InnerProductSIMD8Ext_SSE);
        }
    } break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch
#endif // __x86_64__ */
    return ret_dist_func;
}

} // namespace spaces

#undef X
