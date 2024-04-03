/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#if defined(__x86_64__)
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#endif

namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                       unsigned char *alignment) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = FP32_InnerProduct;
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
        ret_dist_func = Choose_FP32_IP_implementation_AVX512(dim);
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 16 floats
        break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
        ret_dist_func = Choose_FP32_IP_implementation_AVX(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        break;
#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
        ret_dist_func = Choose_FP32_IP_implementation_SSE(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(float); // handles 4 floats
        break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                        unsigned char *alignment) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<double> ret_dist_func = FP64_InnerProduct;
    // Optimizations assume at least 8 doubles. If we have less, we use the naive implementation.
    if (dim < 8) {
        return ret_dist_func;
    }
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
        ret_dist_func = Choose_FP64_IP_implementation_AVX512(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
        ret_dist_func = Choose_FP64_IP_implementation_AVX(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 4 doubles
        break;
#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
        ret_dist_func = Choose_FP64_IP_implementation_SSE(dim);
        if (dim % 2 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 2 * sizeof(double); // handles 2 doubles
        break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__ */
    return ret_dist_func;
}

} // namespace spaces
