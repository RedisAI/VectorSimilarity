/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "SVE_BF16.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/IP/IP_SVE_BF16.h"
#include "VecSim/spaces/L2/L2_SVE_BF16.h"

namespace spaces {

dist_func_t<float> Choose_BF16_IP_implementation_SVE_BF16(size_t dim) {
    return dispatch_by_sve_residual<dist_func_t<float>>(
        dim, svcnth(), []<bool partial_chunk, size_t additional_steps>() {
            return BF16_InnerProduct_SVE<partial_chunk, additional_steps>;
        });
}
dist_func_t<float> Choose_BF16_L2_implementation_SVE_BF16(size_t dim) {
    return dispatch_by_sve_residual<dist_func_t<float>>(
        dim, svcnth(), []<bool partial_chunk, size_t additional_steps>() {
            return BF16_L2Sqr_SVE<partial_chunk, additional_steps>;
        });
}

} // namespace spaces
