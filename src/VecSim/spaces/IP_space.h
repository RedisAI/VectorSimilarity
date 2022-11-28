/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/spaces/spaces.h"

namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt);
dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt);

} // namespace spaces
