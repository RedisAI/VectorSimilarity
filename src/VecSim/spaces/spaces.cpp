/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
namespace spaces {

/*** Defined in spaces.h
 //optimization are defined according to
 // num_bits_per_iteration / bits_count_per_element
    NO_OPTIMIZATION = 0,
    SPLIT_TO_512_BITS = 1,          // FP32 -> dim % 16 == 0, FP64 -> dim % 8 == 0
    SPLIT_TO_512_128_BITS = 2,           // FP32 -> dim % 4 == 0, FP64 -> dim % 2 == 0
    SPLIT_TO_512_BITS_WITH_RESIDUALS = 3, //FP32 ->  dim > 16 && dim % 16 < 4, FP64 -> dim > 8 &&
dim % 8 < 2, SPLIT_TO_512_128_BITS_WITH_RESIDUALS = 4,  // FP32 ->dim > 4, FP64 -> dim > 2
***/
CalculationGuideline FP32_GetCalculationGuideline(size_t dim) {

    CalculationGuideline ret_score = NO_OPTIMIZATION;

    if (dim % 16 == 0) {
        ret_score = SPLIT_TO_512_BITS;
    } else if (dim % 4 == 0) {
        ret_score = SPLIT_TO_512_128_BITS;
    } else if (dim > 16 && dim % 16 < 4) {
        ret_score = SPLIT_TO_512_BITS_WITH_RESIDUALS;
    } else if (dim > 4) {
        ret_score = SPLIT_TO_512_128_BITS_WITH_RESIDUALS;
    }
    return ret_score;
}

CalculationGuideline FP64_GetCalculationGuideline(size_t dim) {

    CalculationGuideline ret_score = NO_OPTIMIZATION;

    if (dim % 8 == 0) {
        ret_score = SPLIT_TO_512_BITS;
    } else if (dim % 2 == 0) {
        ret_score = SPLIT_TO_512_128_BITS;
    } else if (dim > 8 && dim % 8 < 2) {
        ret_score = SPLIT_TO_512_BITS_WITH_RESIDUALS;
    } else if (dim > 2) {
        ret_score = SPLIT_TO_512_128_BITS_WITH_RESIDUALS;
    }
    return ret_score;
}

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {

        *index_dist_func = IP_FP32_GetDistFunc(dim, arch_opt);

    } else if (metric == VecSimMetric_L2) {

        *index_dist_func = L2_FP32_GetDistFunc(dim, arch_opt);
    }
}

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *index_dist_func) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {

        *index_dist_func = IP_FP64_GetDistFunc(dim, arch_opt);

    } else if (metric == VecSimMetric_L2) {

        *index_dist_func = L2_FP64_GetDistFunc(dim, arch_opt);
    }
}

} // namespace spaces
