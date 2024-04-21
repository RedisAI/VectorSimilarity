/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/spaces/functions/AVX512.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/SSE3.h"

using bfloat16 = vecsim_types::bfloat16;

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
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = arch_opt.features;
    if (features.avx512f) {
#ifdef OPT_AVX512F
        ret_dist_func = Choose_FP32_IP_implementation_AVX512(dim);
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 16 floats
        return ret_dist_func;
#endif
    }
    if (features.avx) {
#ifdef OPT_AVX
        ret_dist_func = Choose_FP32_IP_implementation_AVX(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        return ret_dist_func;
#endif
    }
    if (features.sse) {
#ifdef OPT_SSE
        ret_dist_func = Choose_FP32_IP_implementation_SSE(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(float); // handles 4 floats
        return ret_dist_func;
#endif
    }
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
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = arch_opt.features;
    if (features.avx512f) {
#ifdef OPT_AVX512F
        ret_dist_func = Choose_FP64_IP_implementation_AVX512(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        return ret_dist_func;
#endif
    }
    if (features.avx) {
#ifdef OPT_AVX
        ret_dist_func = Choose_FP64_IP_implementation_AVX(dim);
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 4 doubles
        return ret_dist_func;
#endif
    }
    if (features.sse) {
#ifdef OPT_SSE
        ret_dist_func = Choose_FP64_IP_implementation_SSE(dim);
        if (dim % 2 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 2 * sizeof(double); // handles 2 doubles
        return ret_dist_func;
#endif
    }
#endif // __x86_64__ */
    return ret_dist_func;
}

dist_func_t<float> IP_BF16_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                       unsigned char *alignment) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = BF16_InnerProduct_LittleEndian;
    if (!is_little_endian()) {
        return BF16_InnerProduct_BigEndian;
    }
    // Optimizations assume at least 32 bfloats. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = arch_opt.features;
    if (features.avx512bw && features.avx512vbmi2) {
#ifdef OPT_AVX512_BW_VBMI2
        ret_dist_func = Choose_BF16_IP_implementation_AVX512BW_VBMI2(dim);
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(bfloat16); // align to 512 bits.
        return ret_dist_func;
#endif
    }
    if (features.avx2) {
#ifdef OPT_AVX2
        ret_dist_func = Choose_BF16_IP_implementation_AVX2(dim);
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(bfloat16); // align to 256 bits.
        return ret_dist_func;
#endif
    }
    if (features.sse3) {
#ifdef OPT_SSE3
        ret_dist_func = Choose_BF16_IP_implementation_SSE3(dim);
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(bfloat16); // align to 128 bits.
        return ret_dist_func;
#endif
    }
#endif // __x86_64__
    return ret_dist_func;
}

} // namespace spaces
