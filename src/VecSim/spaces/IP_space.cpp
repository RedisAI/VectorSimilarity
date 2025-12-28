/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512BF16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/AVX2_FMA.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/SSE4.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/NEON_DOTPROD.h"
#include "VecSim/spaces/functions/NEON_HP.h"
#include "VecSim/spaces/functions/NEON_BF16.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE_BF16.h"
#include "VecSim/spaces/functions/SVE2.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace spaces {
dist_func_t<float> IP_SQ8_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = SQ8_InnerProduct;
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);
#ifdef CPU_FEATURES_ARCH_AARCH64

#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_SQ8_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_SQ8_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_SQ8_IP_implementation_NEON(dim);
    }
#endif

#endif

#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vnni) {
        return Choose_SQ8_IP_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#ifdef OPT_AVX2_FMA
    if (features.avx2 && features.fma3) {
        return Choose_SQ8_IP_implementation_AVX2_FMA(dim);
    }
#endif
#ifdef OPT_AVX2
    if (features.avx2) {
        return Choose_SQ8_IP_implementation_AVX2(dim);
    }
#endif
#ifdef OPT_SSE4
    if (features.sse4_1) {
        return Choose_SQ8_IP_implementation_SSE4(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> Cosine_SQ8_GetDistFunc(size_t dim, unsigned char *alignment,
                                          const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = SQ8_Cosine;
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);
#ifdef CPU_FEATURES_ARCH_AARCH64

#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_SQ8_Cosine_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_SQ8_Cosine_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_SQ8_Cosine_implementation_NEON(dim);
    }
#endif

#endif

#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vnni) {
        return Choose_SQ8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#ifdef OPT_AVX2_FMA
    if (features.avx2 && features.fma3) {
        return Choose_SQ8_Cosine_implementation_AVX2_FMA(dim);
    }
#endif
#ifdef OPT_AVX2
    if (features.avx2) {
        return Choose_SQ8_Cosine_implementation_AVX2(dim);
    }
#endif
#ifdef OPT_SSE4
    if (features.sse4_1) {
        return Choose_SQ8_Cosine_implementation_SSE4(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = FP32_InnerProduct;
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);
#ifdef CPU_FEATURES_ARCH_AARCH64

#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_FP32_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_FP32_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_FP32_IP_implementation_NEON(dim);
    }
#endif

#endif

#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 16 floats
        return Choose_FP32_IP_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        return Choose_FP32_IP_implementation_AVX(dim);
    }
#endif
#ifdef OPT_SSE
    if (features.sse) {
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(float); // handles 4 floats
        return Choose_FP32_IP_implementation_SSE(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<double> ret_dist_func = FP64_InnerProduct;
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_FP64_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_FP64_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_FP64_IP_implementation_NEON(dim);
    }
#endif

#endif

#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 8 doubles. If we have less, we use the naive implementation.
    if (dim < 8) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        return Choose_FP64_IP_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx) {
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 4 doubles
        return Choose_FP64_IP_implementation_AVX(dim);
    }
#endif
#ifdef OPT_SSE
    if (features.sse) {
        if (dim % 2 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 2 * sizeof(double); // handles 2 doubles
        return Choose_FP64_IP_implementation_SSE(dim);
    }
#endif
#endif // __x86_64__ */
    return ret_dist_func;
}

dist_func_t<float> IP_BF16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = BF16_InnerProduct_LittleEndian;
    if (!is_little_endian()) {
        return BF16_InnerProduct_BigEndian;
    }
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#if defined(CPU_FEATURES_ARCH_AARCH64)
#ifdef OPT_SVE_BF16
    if (features.svebf16) {
        return Choose_BF16_IP_implementation_SVE_BF16(dim);
    }
#endif
#ifdef OPT_NEON_BF16
    if (features.bf16 && dim >= 8) { // Optimization assumes at least 8 BF16s (full chunk)
        return Choose_BF16_IP_implementation_NEON_BF16(dim);
    }
#endif
#endif // AARCH64

#if defined(CPU_FEATURES_ARCH_X86_64)
    // Optimizations assume at least 32 bfloats. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }

#ifdef OPT_AVX512_BF16_VL
    if (features.avx512_bf16 && features.avx512vl) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(bfloat16); // align to 512 bits.
        return Choose_BF16_IP_implementation_AVX512BF16_VL(dim);
    }
#endif
#ifdef OPT_AVX512_BW_VBMI2
    if (features.avx512bw && features.avx512vbmi2) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(bfloat16); // align to 512 bits.
        return Choose_BF16_IP_implementation_AVX512BW_VBMI2(dim);
    }
#endif
#ifdef OPT_AVX2
    if (features.avx2) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(bfloat16); // align to 256 bits.
        return Choose_BF16_IP_implementation_AVX2(dim);
    }
#endif
#ifdef OPT_SSE3
    if (features.sse3) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(bfloat16); // align to 128 bits.
        return Choose_BF16_IP_implementation_SSE3(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> IP_FP16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

    dist_func_t<float> ret_dist_func = FP16_InnerProduct;

#if defined(CPU_FEATURES_ARCH_AARCH64)
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_FP16_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_FP16_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON_HP
    if (features.asimdhp && dim >= 8) { // Optimization assumes at least 8 16FPs (full chunk)
        return Choose_FP16_IP_implementation_NEON_HP(dim);
    }
#endif
#endif

#if defined(CPU_FEATURES_ARCH_X86_64)
    // Optimizations assume at least 32 16FPs. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_FP16_VL
    // More details about the dimension limitation can be found in this PR's description:
    // https://github.com/RedisAI/VectorSimilarity/pull/477
    if (features.avx512_fp16 && features.avx512vl) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(float16); // handles 32 floats
        return Choose_FP16_IP_implementation_AVX512FP16_VL(dim);
    }
#endif
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(float16); // handles 32 floats
        return Choose_FP16_IP_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_F16C
    if (features.f16c && features.fma3 && features.avx) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float16); // handles 16 floats
        return Choose_FP16_IP_implementation_F16C(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> IP_INT8_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = INT8_InnerProduct;

    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_INT8_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_INT8_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON_DOTPROD // Should be the first check, as it is the most optimized
    if (features.asimddp && dim >= 16) {
        return Choose_INT8_IP_implementation_NEON_DOTPROD(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd && dim >= 16) {
        return Choose_INT8_IP_implementation_NEON(dim);
    }
#endif
#endif
#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 32 int8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }

#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(int8_t); // align to 256 bits.
        return Choose_INT8_IP_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> Cosine_INT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = INT8_Cosine;

    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_INT8_Cosine_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_INT8_Cosine_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (features.asimddp && dim >= 16) {
        return Choose_INT8_Cosine_implementation_NEON_DOTPROD(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd && dim >= 16) {
        return Choose_INT8_Cosine_implementation_NEON(dim);
    }
#endif
#endif
#ifdef CPU_FEATURES_ARCH_X86_64
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
        // For int8 vectors with cosine distance, the extra float for the norm shifts alignment to
        // `(dim + sizeof(float)) % 32`.
        // Vectors satisfying this have a residual, causing offset loads during calculation.
        // To avoid complexity, we skip alignment here, assuming the performance impact is
        // negligible.
        return Choose_INT8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif

#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> IP_UINT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = UINT8_InnerProduct;

    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_UINT8_IP_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_UINT8_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (features.asimddp && dim >= 16) {
        return Choose_UINT8_IP_implementation_NEON_DOTPROD(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd && dim >= 16) {
        return Choose_UINT8_IP_implementation_NEON(dim);
    }
#endif
#endif
#ifdef CPU_FEATURES_ARCH_X86_64
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(uint8_t); // align to 256 bits.
        return Choose_UINT8_IP_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> Cosine_UINT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                            const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = UINT8_Cosine;

    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_UINT8_Cosine_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_UINT8_Cosine_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON_DOTPROD
    if (features.asimddp && dim >= 16) {
        return Choose_UINT8_Cosine_implementation_NEON_DOTPROD(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd && dim >= 16) {
        return Choose_UINT8_Cosine_implementation_NEON(dim);
    }
#endif
#endif
#ifdef CPU_FEATURES_ARCH_X86_64
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
        // For uint8 vectors with cosine distance, the extra float for the norm shifts alignment to
        // `(dim + sizeof(float)) % 32`.
        // Vectors satisfying this have a residual, causing offset loads during calculation.
        // To avoid complexity, we skip alignment here, assuming the performance impact is
        // negligible.
        return Choose_UINT8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

// SQ8-to-SQ8 Inner Product distance function (both vectors are uint8 quantized)
dist_func_t<float> IP_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment,
                                          const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = SQ8_SQ8_InnerProduct;
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_SQ8_SQ8_IP_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_SQ8_SQ8_IP_implementation_NEON(dim);
    }
#endif
#endif // AARCH64

#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vnni) {
        return Choose_SQ8_SQ8_IP_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

// SQ8-to-SQ8 Cosine distance function (both vectors are uint8 quantized)
dist_func_t<float> Cosine_SQ8_SQ8_GetDistFunc(size_t dim, unsigned char *alignment,
                                              const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = SQ8_SQ8_Cosine;
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_SQ8_SQ8_Cosine_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_SQ8_SQ8_Cosine_implementation_NEON(dim);
    }
#endif
#endif // AARCH64

#ifdef CPU_FEATURES_ARCH_X86_64
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vnni) {
        return Choose_SQ8_SQ8_Cosine_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

} // namespace spaces
