#pragma once

enum Arch_Optimization { ARCH_OPT_NONE, ARCH_OPT_SSE, ARCH_OPT_AVX, ARCH_OPT_AVX512 };

Arch_Optimization getArchitectureOptimization();
