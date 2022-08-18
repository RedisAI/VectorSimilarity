#pragma once

#include "VecSim/vec_sim_common.h"   // enum VecSimMetric
#include "VecSim/spaces/space_aux.h" //enum  Arch_Optimization
namespace spaces {

static const Arch_Optimization arch_opt = getArchitectureOptimization();

// TODO change last arg to size_t
template <typename RET_TYPE>
using dist_func_t = RET_TYPE (*)(const void *, const void *, const void *dim);

// General optimization logic:
// SIMD16 perform computations on 16 float at a time in each iteration, while SIMD4 perform
// computations on 16 float on most of the vector, and on the residual performing on 4 floats at
// a time.
// When we have a dimension that is not divisible by 4, we should use SIMD16ExtResiduals only if
// the reminder is less than 4, because otherwise we can still perform SIMD4 operations.

enum CalculationGuideline {
    NO_OPTIMIZATION = 0,
    Ext16 = 1,          // dim % 16 == 0
    Ext4 = 2,           // dim % 4 == 0
    ExtResiduals16 = 3, // dim > 16 && dim % 16 < 4
    ExtResiduals4 = 4,  // dim > 4
};

CalculationGuideline GetCalculationGuideline(size_t dim);

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func);

} // namespace spaces
