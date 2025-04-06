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

// General optimization logic:
// SIMD16 perform computations on 16 float at a time in each iteration, while SIMD4 perform
// computations on 16 float on most of the vector, and on the residual performing on 4 floats at
// a time.
// When we have a dimension that is not divisible by 4, we should use SIMD16ExtResiduals only if
// the reminder is less than 4, because otherwise we can still perform SIMD4 operations.

enum CalculationGuideline {
    NO_OPTIMIZATION = 0,
    SPLIT_TO_512_BITS = 1,     // FP32 -> dim % 16 == 0, FP64 -> dim % 8 == 0
    SPLIT_TO_512_128_BITS = 2, // FP32 -> dim % 4 == 0, FP64 -> dim % 2 == 0
    SPLIT_TO_512_BITS_WITH_RESIDUALS =
        3, // FP32 ->  dim > 16 && dim % 16 < 4, FP64 -> dim > 8 && dim % 8 < 2,
    SPLIT_TO_512_128_BITS_WITH_RESIDUALS = 4, // FP32 ->dim > 4, FP64 -> dim > 2
};

CalculationGuideline FP32_GetCalculationGuideline(size_t dim);
CalculationGuideline FP64_GetCalculationGuideline(size_t dim);

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func);
void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *index_dist_func);

} // namespace spaces
