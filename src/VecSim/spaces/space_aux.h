#pragma once

enum Arch_Optimization {
    ARCH_OPT_NONE,
    ARCH_OPT_SSE,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_DQ
};

Arch_Optimization getArchitectureOptimization();
