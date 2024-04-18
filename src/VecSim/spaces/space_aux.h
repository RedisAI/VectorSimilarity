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

namespace spaces {

union Arch_Optimization {
#ifdef CPU_FEATURES_ARCH_X86_64
    cpu_features::X86Features features;
#endif
    char no_opt; // dummy field
};

Arch_Optimization getArchitectureOptimization();

static int inline is_little_endian() {
    unsigned int x = 1;
    return *(char *)&x;
}

} // namespace spaces
