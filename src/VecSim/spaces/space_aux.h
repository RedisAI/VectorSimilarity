/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "cpu_features_macros.h"

namespace spaces {

enum Arch_Optimization {
    ARCH_OPT_NONE,
    ARCH_OPT_SSE,
    ARCH_OPT_SSE3,
    ARCH_OPT_AVX,
    ARCH_OPT_AVX2,
    ARCH_OPT_AVX512_F,
    ARCH_OPT_AVX512_BW_VBMI2,
};

Arch_Optimization getArchitectureOptimization();

static int inline is_little_endian() {
    unsigned int x = 1;
    return *(char *)&x;
}

} // namespace spaces
