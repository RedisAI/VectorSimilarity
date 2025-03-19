/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <armpl.h>

template <unsigned char residual>
double FP64_InnerProduct_ARMPL_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    auto *vec1 = (double *)pVect1v;
    auto *vec2 = (double *)pVect2v;

    double res = cblas_ddot(static_cast<int>(dimension), vec1, 1, vec2, 1);
    return 1.0f - res;
}
