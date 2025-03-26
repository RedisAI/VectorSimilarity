/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/spaces/functions/F16C.h"
#include "VecSim/spaces/functions/AVX512F.h"
#include "VecSim/spaces/functions/AVX.h"
#include "VecSim/spaces/functions/SSE.h"
#include "VecSim/spaces/functions/AVX512BW_VBMI2.h"
#include "VecSim/spaces/functions/AVX512FP16_VL.h"
#include "VecSim/spaces/functions/AVX512F_BW_VL_VNNI.h"
#include "VecSim/spaces/functions/AVX2.h"
#include "VecSim/spaces/functions/SSE3.h"
#include "VecSim/spaces/functions/NEON.h"
#include "VecSim/spaces/functions/SVE.h"
#include "VecSim/spaces/functions/SVE2.h"

using bfloat16 = vecsim_types::bfloat16;
using float16 = vecsim_types::float16;

namespace spaces {

dist_func_t<float> L2_FP32_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = FP32_L2Sqr;
    // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
    if (dim < 16) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_AARCH64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetAarch64Info().features
                        : *static_cast<const cpu_features::Aarch64Features *>(arch_opt);

#ifdef OPT_SVE2
    if (features.sve2) {
        return Choose_FP32_L2_implementation_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_FP32_L2_implementation_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_FP32_L2_implementation_NEON(dim);
    }
#endif
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float); // handles 16 floats
        return Choose_FP32_L2_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(float); // handles 8 floats
        return Choose_FP32_L2_implementation_AVX(dim);
    }
#endif
#ifdef OPT_SSE
    if (features.sse) {
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(float); // handles 4 floats
        return Choose_FP32_L2_implementation_SSE(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> L2_FP64_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
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
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(double); // handles 8 doubles
        return Choose_FP64_L2_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx) {
        if (dim % 4 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 4 * sizeof(double); // handles 4 doubles
        return Choose_FP64_L2_implementation_AVX(dim);
    }
#endif
#ifdef OPT_SSE
    if (features.sse) {
        if (dim % 2 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 2 * sizeof(double); // handles 2 doubles
        return Choose_FP64_L2_implementation_SSE(dim);
    }
#endif
#endif // __x86_64__ */
    return ret_dist_func;
}

dist_func_t<float> L2_BF16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (!alignment) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = BF16_L2Sqr_LittleEndian;
    if (!is_little_endian()) {
        return BF16_L2Sqr_BigEndian;
    }
    // Optimizations assume at least 32 bfloats. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512_BW_VBMI2
    if (features.avx512bw && features.avx512vbmi2) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(bfloat16); // align to 512 bits.
        return Choose_BF16_L2_implementation_AVX512BW_VBMI2(dim);
    }
#endif
#ifdef OPT_AVX2
    if (features.avx2) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(bfloat16); // align to 256 bits.
        return Choose_BF16_L2_implementation_AVX2(dim);
    }
#endif
#ifdef OPT_SSE3
    if (features.sse3) {
        if (dim % 8 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 8 * sizeof(bfloat16); // align to 128 bits.
        return Choose_BF16_L2_implementation_SSE3(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> L2_FP16_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = FP16_L2Sqr;
    // Optimizations assume at least 32 16FPs. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512_FP16_VL
    // More details about the dimension limitation can be found in this PR's description:
    // https://github.com/RedisAI/VectorSimilarity/pull/477
    if (features.avx512_fp16 && features.avx512vl) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(float16); // handles 32 floats
        return Choose_FP16_L2_implementation_AVX512FP16_VL(dim);
    }
#endif
#ifdef OPT_AVX512F
    if (features.avx512f) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(float16); // handles 32 floats
        return Choose_FP16_L2_implementation_AVX512F(dim);
    }
#endif
#ifdef OPT_F16C
    if (features.f16c && features.fma3 && features.avx) {
        if (dim % 16 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 16 * sizeof(float16); // handles 16 floats
        return Choose_FP16_L2_implementation_F16C(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> L2_INT8_GetDistFunc(size_t dim, unsigned char *alignment, const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = INT8_L2Sqr;
    // Optimizations assume at least 32 int8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(int8_t); // align to 256 bits.
        return Choose_INT8_L2_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve) {
        return Choose_INT8_L2_implementation_SVE(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<float> L2_UINT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                        const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = UINT8_L2Sqr;
    // Optimizations assume at least 32 uint8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
#ifdef CPU_FEATURES_ARCH_X86_64
    auto features = (arch_opt == nullptr)
                        ? cpu_features::GetX86Info().features
                        : *static_cast<const cpu_features::X86Features *>(arch_opt);
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (features.avx512f && features.avx512bw && features.avx512vl && features.avx512vnni) {
        if (dim % 32 == 0) // no point in aligning if we have an offsetting residual
            *alignment = 32 * sizeof(int8_t); // align to 256 bits.
        return Choose_UINT8_L2_implementation_AVX512F_BW_VL_VNNI(dim);
    }
#endif
#endif // __x86_64__
    return ret_dist_func;
}

} // namespace spaces
