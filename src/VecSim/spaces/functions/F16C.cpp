/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "F16C.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/IP/IP_F16C_FP16.h"
#include "VecSim/spaces/L2/L2_F16C_FP16.h"

namespace spaces {

dist_func_t<float> Choose_FP16_IP_implementation_F16C(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP16_InnerProductSIMD32_F16C<N>; });
}

dist_func_t<float> Choose_FP16_L2_implementation_F16C(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return FP16_L2SqrSIMD32_F16C<N>; });
}

} // namespace spaces
