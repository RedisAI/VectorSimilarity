/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "space_aux.h"

Arch_Optimization getArchitectureOptimization() {

#ifdef CPU_FEATURES_ARCH_X86_64
    cpu_features::X86Features features = cpu_features::GetX86Info().features;
    if (features.avx512dq) {
        return ARCH_OPT_AVX512_DQ;
    }
    if (features.avx512f) {
        return ARCH_OPT_AVX512_F;
    } else if (features.avx || features.avx2) {
        return ARCH_OPT_AVX;
    } else if (features.sse || features.sse2 || features.sse3 || features.sse4_1 ||
               features.sse4_2 || features.sse4a) {
        return ARCH_OPT_SSE;
    }
#endif // CPU_FEATURES_ARCH_X86_64
#ifdef CPU_FEATURES_ARCH_AARCH64
    cpu_features::Aarch64Features features = cpu_features::GetAarch64Info().features;
    if (features.sve2) {
        return ARCH_OPT_SVE2;
    }
    if (features.sve) {
        return ARCH_OPT_SVE;
    }
    if (features.asimd) {
        return ARCH_OPT_NEON;
    }
#endif // CPU_FEATURES_ARCH_AARCH64

    return ARCH_OPT_NONE;
}
