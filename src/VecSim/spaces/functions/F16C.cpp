/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "F16C.h"

#include "VecSim/spaces/IP/IP_F16C_FP16.h"
#include "VecSim/spaces/L2/L2_F16C_FP16.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_FP16_IP_implementation_F16C(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP16_InnerProductSIMD16_F16C);
    return ret_dist_func;
}

dist_func_t<float> Choose_FP16_L2_implementation_F16C(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 32, FP16_L2SqrSIMD16_F16C);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
