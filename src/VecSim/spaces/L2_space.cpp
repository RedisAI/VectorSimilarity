/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/types/bfloat16.h"
#if defined(__x86_64__)
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/SSE3.h"
#endif

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
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_BW_VBMI2:
    case ARCH_OPT_AVX512_F:
#ifdef OPT_AVX512F
        ret_dist_func = Choose_FP32_L2_implementation_AVX512(dim);
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 16 floats
        break;
#endif
    case ARCH_OPT_AVX2:
    case ARCH_OPT_AVX:
#ifdef OPT_AVX
        ret_dist_func = Choose_FP32_L2_implementation_AVX(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        break;
#endif
    case ARCH_OPT_SSE3:
    case ARCH_OPT_SSE:
#ifdef OPT_SSE
        ret_dist_func = Choose_FP32_L2_implementation_SSE(dim);
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
#if defined(M1)
#elif defined(__x86_64__)

    switch (arch_opt) {
    case ARCH_OPT_AVX512_BW_VBMI2:
    case ARCH_OPT_AVX512_F:
#ifdef OPT_AVX512F
        ret_dist_func = Choose_FP64_L2_implementation_AVX512(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        break;
#endif
    case ARCH_OPT_AVX2:
    case ARCH_OPT_AVX:
#ifdef OPT_AVX
        ret_dist_func = Choose_FP64_L2_implementation_AVX(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 4 doubles
        break;
#endif
    case ARCH_OPT_SSE3:
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

dist_func_t<float> L2_BF16_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                       unsigned char *alignment) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = BF16_L2Sqr_LittleEndian;
    if (!is_little_endian()) {
        ret_dist_func = BF16_L2Sqr_BigEndian;
    }
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#if defined(M1)
#elif defined(__x86_64__)
    switch (arch_opt) {
    case ARCH_OPT_AVX512_BW_VBMI2:
#ifdef OPT_AVX512_BW_VBMI2
        ret_dist_func = Choose_BF16_L2_implementation_AVX512BW_VBMI2(dim);
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(bfloat16); // align to 512 bits.
        break;
#endif
    case ARCH_OPT_AVX512_F:
    case ARCH_OPT_AVX2:
#ifdef OPT_AVX2
        ret_dist_func = Choose_BF16_L2_implementation_AVX2(dim);
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(bfloat16); // align to 256 bits.
        break;
#endif
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE3:
#ifdef OPT_SSE3
        ret_dist_func = Choose_BF16_L2_implementation_SSE3(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(bfloat16); // align to 128 bits.
        break;
#endif
    case ARCH_OPT_SSE:
    case ARCH_OPT_NONE:
        break;
    } // switch

#endif // __x86_64__
    return ret_dist_func;
}

} // namespace spaces
