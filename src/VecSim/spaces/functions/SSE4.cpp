/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "SSE4.h"
#include "VecSim/spaces/IP/IP_SSE4_SQ8.h"
#include "VecSim/spaces/L2/L2_SSE4_SQ8.h"

namespace spaces {

#include "implementation_chooser.h"

dist_func_t<float> Choose_SQ8_IP_implementation_SSE4(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, SQ8_InnerProductSIMD16_SSE4);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_Cosine_implementation_SSE4(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, SQ8_CosineSIMD16_SSE4);
    return ret_dist_func;
}

dist_func_t<float> Choose_SQ8_L2_implementation_SSE4(size_t dim) {
    dist_func_t<float> ret_dist_func;
    CHOOSE_IMPLEMENTATION(ret_dist_func, dim, 16, SQ8_L2SqrSIMD16_SSE4);
    return ret_dist_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
