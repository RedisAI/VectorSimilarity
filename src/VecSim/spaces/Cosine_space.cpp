/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "VecSim/spaces/Cosine_space.h"
#include "VecSim/spaces/Cosine/Cosine.h"

namespace spaces {
dist_func_t<float> Cosine_INT8_GetDistFunc(size_t dim, unsigned char *alignment,
                                           const void *arch_opt) {
    unsigned char dummy_alignment;
    if (alignment == nullptr) {
        alignment = &dummy_alignment;
    }

    dist_func_t<float> ret_dist_func = INT8_Cosine;
    // Optimizations assume at least 32 int8. If we have less, we use the naive implementation.
    if (dim < 32) {
        return ret_dist_func;
    }
    return ret_dist_func;
}

} // namespace spaces
