/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "AVX512BF16_VL.h"
#include "VecSim/spaces/functions/residual_dispatch.h"

#include "VecSim/spaces/IP/IP_AVX512_BF16_VL_BF16.h"

namespace spaces {

dist_func_t<float> Choose_BF16_IP_implementation_AVX512BF16_VL(size_t dim) {
    return dispatch_by_residual<dist_func_t<float>, 32>(
        dim, []<size_t N>() { return BF16_InnerProductSIMD32_AVX512BF16_VL<N>; });
}

} // namespace spaces
