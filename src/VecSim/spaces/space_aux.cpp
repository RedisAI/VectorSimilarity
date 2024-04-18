/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "space_aux.h"

namespace spaces {

Arch_Optimization getArchitectureOptimization() {

#ifdef CPU_FEATURES_ARCH_X86_64
    cpu_features::X86Features features = cpu_features::GetX86Info().features;
    return Arch_Optimization{.features = features};
#endif // CPU_FEATURES_ARCH_X86_64

    return Arch_Optimization{.no_opt = 0};
}

} // namespace spaces
