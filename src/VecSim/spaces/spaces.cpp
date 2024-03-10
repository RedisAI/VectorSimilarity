/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/encoders/bf16/BF16_encoder_impl.h"
#include "VecSim/spaces/IP/IP.h"
namespace spaces {

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *out_func,
                 unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {

        *out_func = IP_FP32_GetDistFunc(dim, arch_opt, alignment);

    } else if (metric == VecSimMetric_L2) {

        *out_func = L2_FP32_GetDistFunc(dim, arch_opt, alignment);
    }
}

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *out_func,
                 unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {

        *out_func = IP_FP64_GetDistFunc(dim, arch_opt, alignment);

    } else if (metric == VecSimMetric_L2) {

        *out_func = L2_FP64_GetDistFunc(dim, arch_opt, alignment);
    }
}

int little_endian() {
    int x = 1;
    return *(char *)&x;
}
int big_endian() { return !little_endian(); }

// fp32_to_bf16_encoder_t GetFP32ToBFloat16Encoder(size_t dim) {

//     static const Arch_Optimization arch_opt = getArchitectureOptimization();
//     return Get_FP32_to_BF16_Encoder(dim, arch_opt, big_endian());
// }

void SetBF16DistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *out_func,
                     unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {
        if (little_endian()) {
            *out_func = IP_FP32_GetDistFunc(dim, arch_opt, alignment, VecSimType_FP32_TO_BF16);

        } else {
            *out_func = BFP16_InnerProduct;
        }

    } else if (metric == VecSimMetric_L2) {
        /* not supported yet */
        *out_func = NULL;
    }
}
void SetBF16DistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *out_func,
                     unsigned char *alignment) {
    *out_func = NULL;
}

void SetFP16DistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *out_func,
                     unsigned char *alignment) {

    static const Arch_Optimization arch_opt = getArchitectureOptimization();

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {
        *out_func = IP_FP32_GetDistFunc(dim, arch_opt, alignment, VecSimType_FP32_TO_FP16);

    } else if (metric == VecSimMetric_L2) {
        /* not supported yet */
        *out_func = NULL;
    }
}
void SetFP16DistFunc(VecSimMetric metric, size_t dim, dist_func_t<double> *out_func,
                     unsigned char *alignment) {
    *out_func = NULL;
}
} // namespace spaces
