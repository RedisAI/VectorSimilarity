/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE2.h"

namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<float> ret_dist_func = FP32_InnerProduct;
    auto features = getCpuOptimizationFeatures(arch_opt);
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

#ifdef CPU_FEATURES_ARCH_X86_64
    CalculationGuideline optimization_type = FP32_GetCalculationGuideline(dim);
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#endif
    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:

#ifdef OPT_AVX512F
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_AVX512, FP32_InnerProductSIMD4Ext_AVX512,
            FP32_InnerProductSIMD16ExtResiduals_AVX512, FP32_InnerProductSIMD4ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef OPT_AVX
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_AVX, FP32_InnerProductSIMD4Ext_AVX,
            FP32_InnerProductSIMD16ExtResiduals_AVX, FP32_InnerProductSIMD4ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef OPT_SSE
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_SSE, FP32_InnerProductSIMD4Ext_SSE,
            FP32_InnerProductSIMD16ExtResiduals_SSE, FP32_InnerProductSIMD4ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch
#endif // __x86_64__
#ifdef CPU_FEATURES_ARCH_AARCH64
    case ARCH_OPT_SVE2:
#ifdef OPT_SVE2
        ret_dist_func = Choose_FP32_IP_implementation_SVE2(dim);
        break;
#endif
    case ARCH_OPT_SVE:
#ifdef OPT_SVE
        ret_dist_func = Choose_FP32_IP_implementation_SVE(dim);
        break;
#endif
    case ARCH_OPT_NEON:
#ifdef OPT_NEON
        ret_dist_func = Choose_FP32_IP_implementation_NEON(dim);
        break;
#endif

#endif // CPU_FEATURES_ARCH_X86_64
    case ARCH_OPT_NONE:
        break;
    } // switch
    return ret_dist_func;
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<double> ret_dist_func = FP64_InnerProduct;
#ifdef CPU_FEATURES_ARCH_X86_64

    CalculationGuideline optimization_type = FP64_GetCalculationGuideline(dim);

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
#ifdef OPT_AVX512DQ
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX512, FP64_InnerProductSIMD2Ext_AVX512,
            FP64_InnerProductSIMD8ExtResiduals_AVX512, FP64_InnerProductSIMD2ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX512_F:
#ifdef OPT_AVX512F
    {
        // If AVX512 foundation flag is supported, but AVX512DQ isn't supported, we cannot extract
        // 2X64-bit elements from the 512bit register, which is required when dim%8 != 0, so we can
        // continue the vector computations by using 128 register optimization on the vectors'
        // tails. Then, we use modified versions that split both part of the computation without
        // using the unsupported extraction operation.
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX512,
            FP64_InnerProductSIMD8ExtResiduals_AVX512, FP64_InnerProductSIMD2Ext_AVX512_noDQ,
            FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef OPT_AVX
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX, FP64_InnerProductSIMD2Ext_AVX,
            FP64_InnerProductSIMD8ExtResiduals_AVX, FP64_InnerProductSIMD2ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef OPT_SSE
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_SSE, FP64_InnerProductSIMD2Ext_SSE,
            FP64_InnerProductSIMD8ExtResiduals_SSE, FP64_InnerProductSIMD2ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch
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
    // Optimizations assume at least 32 bfloats. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }

#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = getCpuOptimizationFeatures(arch_opt);
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

    dist_func_t<float> ret_dist_func = FP16_InnerProduct;
    // Optimizations assume at least 32 16FPs. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = getCpuOptimizationFeatures(arch_opt);
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
    // Optimizations assume at least 32 int8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = getCpuOptimizationFeatures(arch_opt);
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
    // Optimizations assume at least 32 int8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = getCpuOptimizationFeatures(arch_opt);
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
    // Optimizations assume at least 32 uint8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = getCpuOptimizationFeatures(arch_opt);
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
    // Optimizations assume at least 32 uint8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = getCpuOptimizationFeatures(arch_opt);
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

} // namespace spaces
