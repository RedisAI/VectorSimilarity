#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { ARCH_OPT_NONE, ARCH_OPT_SSE, ARCH_OPT_AVX, ARCH_OPT_AVX512 } Arch_Optimization;

Arch_Optimization getArchitectureOptimization();

#ifdef __cplusplus
}
#endif
