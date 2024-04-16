/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "space_aux.h"

namespace spaces {

#ifdef CPU_FEATURES_ARCH_X86_64
#include "cpuinfo_x86.h"
#endif // CPU_FEATURES_ARCH_X86_64

Arch_Optimization getArchitectureOptimization() {

#ifdef CPU_FEATURES_ARCH_X86_64
    cpu_features::X86Features features = cpu_features::GetX86Info().features;
    if (features.avx512bw && features.avx512vl) {
        return ARCH_OPT_AVX512_BW_VL;
    } else if (features.avx512bw && features.avx512vbmi2) {
        return ARCH_OPT_AVX512_BW_VBMI2;
    } else if (features.avx512f) {
        return ARCH_OPT_AVX512_F;
    } else if (features.avx && features.f16c && features.fma3) {
        return ARCH_OPT_F16C;
    } else if (features.avx2) {
        return ARCH_OPT_AVX2;
    } else if (features.avx) {
        return ARCH_OPT_AVX;
    } else if (features.sse3) {
        return ARCH_OPT_SSE3;
    } else if (features.sse) {
        return ARCH_OPT_SSE;
    }
#endif // CPU_FEATURES_ARCH_X86_64

    return ARCH_OPT_NONE;
}

} // namespace spaces
