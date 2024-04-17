/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/types/bfloat16.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/normalize/normalize_naive.h"
#include "VecSim/spaces/spaces.h"

#include <stdexcept>
namespace spaces {

// Set the distance function for a given data type, metric and dimension. The alignment hint is
// determined according to the chosen implementation and available optimizations.

template <>
dist_func_t<float> GetDistFunc<vecsim_types::bfloat16, float>(VecSimMetric metric, size_t dim,
                                                              unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    switch (metric) {
    case VecSimMetric_Cosine:
    case VecSimMetric_IP:
        return IP_BF16_GetDistFunc(dim, arch_opt, alignment);
    case VecSimMetric_L2:
        return L2_BF16_GetDistFunc(dim, arch_opt, alignment);
    default:
        throw std::invalid_argument("Invalid metric");
    }
}

template <>
dist_func_t<float> GetDistFunc<float, float>(VecSimMetric metric, size_t dim,
                                             unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    switch (metric) {
    case VecSimMetric_Cosine:
    case VecSimMetric_IP:
        return IP_FP32_GetDistFunc(dim, arch_opt, alignment);
    case VecSimMetric_L2:
        return L2_FP32_GetDistFunc(dim, arch_opt, alignment);
    default:
        throw std::invalid_argument("Invalid metric");
    }
}

template <>
dist_func_t<double> GetDistFunc<double, double>(VecSimMetric metric, size_t dim,
                                                unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    switch (metric) {
    case VecSimMetric_Cosine:
    case VecSimMetric_IP:
        return IP_FP64_GetDistFunc(dim, arch_opt, alignment);
    case VecSimMetric_L2:
        return L2_FP64_GetDistFunc(dim, arch_opt, alignment);
    default:
        throw std::invalid_argument("Invalid metric");
    }
}

template <>
normalizeVector_f<float> GetNormalizeFunc<float>(void) {
    return normalizeVector_imp<float>;
}

template <>
normalizeVector_f<double> GetNormalizeFunc<double>(void) {
    return normalizeVector_imp<double>;
}

template <>
normalizeVector_f<vecsim_types::bfloat16> GetNormalizeFunc<vecsim_types::bfloat16>(void) {
    if (is_little_endian()) {
        return bfloat16_normalizeVector<true>;
    } else {
        return bfloat16_normalizeVector<false>;
    }
}
} // namespace spaces
