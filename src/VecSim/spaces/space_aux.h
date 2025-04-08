/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "cpu_features_macros.h"
#ifdef CPU_FEATURES_ARCH_X86_64
#include "cpuinfo_x86.h"
#endif // CPU_FEATURES_ARCH_X86_64
#ifdef CPU_FEATURES_ARCH_AARCH64
#include "cpuinfo_aarch64.h"
#endif // CPU_FEATURES_ARCH_AARCH64

enum Arch_Optimization {
    ARCH_OPT_NONE,
#ifdef CPU_FEATURES_ARCH_X86_64
    ARCH_OPT_SSE,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_DQ,
#endif
#ifdef CPU_FEATURES_ARCH_AARCH64
    ARCH_OPT_NEON,
    ARCH_OPT_SVE,
    ARCH_OPT_SVE2,
#endif
    ARCH_OPT_LENGTH,
};

Arch_Optimization getArchitectureOptimization();
