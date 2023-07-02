/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/vec_sim_common.h"   // enum VecSimMetric
#include "VecSim/spaces/space_aux.h" //enum  Arch_Optimization
namespace spaces {

template <typename RET_TYPE>
using dist_func_t = RET_TYPE (*)(const void *, const void *, size_t);

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func);
void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *index_dist_func);

} // namespace spaces
