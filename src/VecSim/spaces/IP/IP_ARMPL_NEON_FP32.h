/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <armpl.h>

template <unsigned char residual> // 0..15
float FP32_InnerProductSIMD16_ARMPL_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *vec1 = (float *)pVect1v;
    auto *vec2 = (float *)pVect2v;

    float res = cblas_sdot(static_cast<int>(dimension), vec1, 1, vec2, 1);
    return 1.0f - res; 
}