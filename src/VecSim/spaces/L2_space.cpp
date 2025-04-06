/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE2.h"

namespace spaces {

dist_func_t<float> L2_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                       unsigned char *alignment) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = FP32_L2Sqr;

    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.

    if (dim < 16) {
        return ret_dist_func;
    }
    switch (arch_opt) {
#ifdef CPU_FEATURES_ARCH_X86_64

    case ARCH_OPT_AVX512_F:
#ifdef OPT_AVX512F
        ret_dist_func = Choose_FP32_L2_implementation_AVX512(dim);
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 16 floats
        break;
#endif
    case ARCH_OPT_AVX:
#ifdef OPT_AVX
        ret_dist_func = Choose_FP32_L2_implementation_AVX(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        break;
#endif
    case ARCH_OPT_SSE:
#ifdef OPT_SSE
        ret_dist_func = Choose_FP32_L2_implementation_SSE(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(float); // handles 4 floats
        break;
#endif
#endif // __x86_64__
#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    case ARCH_OPT_SVE2:
        ret_dist_func = Choose_FP32_L2_implementation_SVE2(dim);
        break;
    
#endif
#ifdef OPT_SVE
    case ARCH_OPT_SVE:
        ret_dist_func =  Choose_FP32_L2_implementation_SVE(dim);
        break;

    
#endif
#ifdef OPT_NEON
    case ARCH_OPT_NEON:
        ret_dist_func = Choose_FP32_L2_implementation_NEON(dim);
        break;
#endif
#endif // __aarch64__
case ARCH_OPT_NONE:
        break;
} // switch
    return ret_dist_func;
}

dist_func_t<double> L2_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                        unsigned char *alignment) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<double> ret_dist_func = FP64_L2Sqr;
    // Optimizations assume at least 8 doubles. If we have less, we use the naive implementation.
    if (dim < 8) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64

    switch (arch_opt) {
    case ARCH_OPT_AVX512_F:
#ifdef OPT_AVX512F
        ret_dist_func = Choose_FP64_L2_implementation_AVX512(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        break;
#endif
    case ARCH_OPT_AVX:
#ifdef OPT_AVX
        ret_dist_func = Choose_FP64_L2_implementation_AVX(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 4 doubles
        break;
#endif
    case ARCH_OPT_SSE:
#ifdef OPT_SSE
        ret_dist_func = Choose_FP64_L2_implementation_SSE(dim);
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
