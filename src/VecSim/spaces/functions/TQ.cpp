/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "TQ.h"

#ifdef CPU_FEATURES_ARCH_AARCH64
#include "TQ_POLAR_NEON.h"
namespace spaces {
#ifdef OPT_NEON
tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_NEON(size_t dim);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_NEON(size_t dim);
#endif
#ifdef OPT_SVE
tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_SVE(size_t dim);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_SVE(size_t dim);
#endif
#ifdef OPT_SVE2
tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_SVE2(size_t dim);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_SVE2(size_t dim);
#endif
} // namespace spaces
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
#include "TQ_POLAR_AVX2.h"
#include "TQ_POLAR_AVX512F_BW_VL_VNNI.h"
#include "TQ_POLAR_SSE4.h"
namespace spaces {
#ifdef OPT_SSE
tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_SSE(size_t dim);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_SSE(size_t dim);
#endif
#ifdef OPT_AVX
tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_AVX(size_t dim);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_AVX(size_t dim);
#endif
#ifdef OPT_AVX512F
tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_AVX512F(size_t dim);
tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_AVX512F(size_t dim);
#endif
} // namespace spaces
#endif

namespace spaces {

namespace {

inline uint16_t AngleCodeAt(const void *angles, size_t idx, bool nibble_angle_codes,
                            bool compact_angle_codes) {
    if (nibble_angle_codes) {
        const auto *encoded = static_cast<const uint8_t *>(angles);
        const uint8_t packed = encoded[idx / 2];
        return (idx % 2 == 0) ? static_cast<uint16_t>(packed & 0x0F)
                              : static_cast<uint16_t>((packed >> 4) & 0x0F);
    }
    if (compact_angle_codes) {
        return static_cast<const uint8_t *>(angles)[idx];
    }
    return static_cast<const uint16_t *>(angles)[idx];
}

} // namespace

tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ(size_t dim,
                                                                   const void *arch_opt) {
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2 && dim >= 16) {
        return Choose_FP32_InnerProduct_implementation_TQ_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve && dim >= 16) {
        return Choose_FP32_InnerProduct_implementation_TQ_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd && dim >= 16) {
        return Choose_FP32_InnerProduct_implementation_TQ_NEON(dim);
    }
#endif
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
#ifdef OPT_AVX512F
    if (features.avx512f && dim >= 16) {
        return Choose_FP32_InnerProduct_implementation_TQ_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx && dim >= 16) {
        return Choose_FP32_InnerProduct_implementation_TQ_AVX(dim);
    }
#endif
#ifdef OPT_SSE
    if (features.sse && dim >= 16) {
        return Choose_FP32_InnerProduct_implementation_TQ_SSE(dim);
    }
#endif
#endif

    return nullptr;
}

tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ(size_t dim, const void *arch_opt) {
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_SVE2
    if (features.sve2 && dim >= 16) {
        return Choose_FP32_SumSquares_implementation_TQ_SVE2(dim);
    }
#endif
#ifdef OPT_SVE
    if (features.sve && dim >= 16) {
        return Choose_FP32_SumSquares_implementation_TQ_SVE(dim);
    }
#endif
#ifdef OPT_NEON
    if (features.asimd && dim >= 16) {
        return Choose_FP32_SumSquares_implementation_TQ_NEON(dim);
    }
#endif
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
#ifdef OPT_AVX512F
    if (features.avx512f && dim >= 16) {
        return Choose_FP32_SumSquares_implementation_TQ_AVX512F(dim);
    }
#endif
#ifdef OPT_AVX
    if (features.avx && dim >= 16) {
        return Choose_FP32_SumSquares_implementation_TQ_AVX(dim);
    }
#endif
#ifdef OPT_SSE
    if (features.sse && dim >= 16) {
        return Choose_FP32_SumSquares_implementation_TQ_SSE(dim);
    }
#endif
#endif

    return nullptr;
}

tq_symmetric_polar_estimate_func_t
Choose_TQ_SymmetricPolarEstimate_implementation(const void *arch_opt) {
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_TQ_SymmetricPolarEstimate_implementation_NEON();
    }
#endif
#endif

    return TQ_SymmetricPolarEstimate;
}

tq_packed_sign_dot_func_t Choose_TQ_PackedSignDot_implementation(const void *arch_opt) {
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_NEON
    if (features.asimd) {
        return Choose_TQ_PackedSignDot_implementation_NEON();
    }
#endif
#endif

    return TQ_PackedSignDot;
}

tq_packed_residual_sign_dot_func_t
Choose_TQ_PackedResidualSignDot_implementation(size_t projections, const void *arch_opt) {
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);
    const size_t full_bytes = projections / 8;

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_NEON
    if (features.asimd && full_bytes >= 16) {
        return Choose_TQ_PackedResidualSignDot_implementation_NEON(projections);
    }
#endif
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (full_bytes >= 64 && features.avx512f && features.avx512bw && features.avx512vnni) {
        return Choose_TQ_PackedResidualSignDot_implementation_AVX512F_BW_VL_VNNI(projections);
    }
#endif
#ifdef OPT_AVX2
    if (full_bytes >= 32 && features.avx2) {
        return Choose_TQ_PackedResidualSignDot_implementation_AVX2(projections);
    }
#endif
#ifdef OPT_SSE4
    if (full_bytes >= 16 && features.sse4_1) {
        return Choose_TQ_PackedResidualSignDot_implementation_SSE4(projections);
    }
#endif
#endif

    return nullptr;
}

tq_symmetric_polar_func_t Choose_TQ_SymmetricPolar_implementation(size_t pairs,
                                                                  const void *arch_opt) {
    [[maybe_unused]] auto features = getCpuOptimizationFeatures(arch_opt);

#ifdef CPU_FEATURES_ARCH_AARCH64
#ifdef OPT_NEON
    if (features.asimd && pairs >= 8) {
        return Choose_TQ_SymmetricPolar_implementation_NEON(pairs);
    }
#endif
#endif

#ifdef CPU_FEATURES_ARCH_X86_64
#ifdef OPT_AVX512_F_BW_VL_VNNI
    if (pairs >= 16 && features.avx512f && features.avx512bw && features.avx512vnni) {
        return Choose_TQ_SymmetricPolar_implementation_AVX512F_BW_VL_VNNI(pairs);
    }
#endif
#ifdef OPT_AVX2
    if (pairs >= 8 && features.avx2) {
        return Choose_TQ_SymmetricPolar_implementation_AVX2(pairs);
    }
#endif
#ifdef OPT_SSE4
    if (pairs >= 4 && features.sse4_1) {
        return Choose_TQ_SymmetricPolar_implementation_SSE4(pairs);
    }
#endif
#endif

    return nullptr;
}

float TQ_SymmetricPolarEstimate(const float *lhs_radii, const void *lhs_angles,
                                const float *rhs_radii, const void *rhs_angles, size_t pairs,
                                size_t angle_delta_mask, const float *delta_cos_lut,
                                bool nibble_angle_codes, bool compact_angle_codes) {
    float polar_estimate = 0.0f;
    for (size_t idx = 0; idx < pairs; ++idx) {
        const uint16_t lhs_angle =
            AngleCodeAt(lhs_angles, idx, nibble_angle_codes, compact_angle_codes);
        const uint16_t rhs_angle =
            AngleCodeAt(rhs_angles, idx, nibble_angle_codes, compact_angle_codes);
        const size_t delta =
            (static_cast<size_t>(lhs_angle) - static_cast<size_t>(rhs_angle)) & angle_delta_mask;
        polar_estimate += lhs_radii[idx] * rhs_radii[idx] * delta_cos_lut[delta];
    }
    return polar_estimate;
}

int TQ_PackedSignDot(const uint8_t *lhs, const uint8_t *rhs, size_t projections) {
    const size_t full_bytes = projections / 8;
    const size_t tail_bits = projections % 8;
    int sign_dot = 0;

    for (size_t idx = 0; idx < full_bytes; ++idx) {
        const int diff_count = __builtin_popcount(static_cast<unsigned int>(lhs[idx] ^ rhs[idx]));
        sign_dot += 8 - (2 * diff_count);
    }

    if (tail_bits != 0) {
        const uint8_t valid_mask = static_cast<uint8_t>((uint16_t{1} << tail_bits) - 1u);
        const uint8_t diff_bits =
            static_cast<uint8_t>((lhs[full_bytes] ^ rhs[full_bytes]) & valid_mask);
        const int diff_count = __builtin_popcount(static_cast<unsigned int>(diff_bits));
        sign_dot += static_cast<int>(tail_bits) - (2 * diff_count);
    }

    return sign_dot;
}

} // namespace spaces
