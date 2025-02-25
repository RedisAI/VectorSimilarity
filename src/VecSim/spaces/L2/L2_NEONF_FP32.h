/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

 #include "VecSim/spaces/space_includes.h"
#ifdef OPT_NEON
 #include <armpl.h>
#endif

template <unsigned char residual> // 0..15
float FP32_L2SqrSIMD16_NEONF(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *vec1 = static_cast<const float*>(pVect1v);
    const float *vec2 = static_cast<const float*>(pVect2v);

    float dot_xx = cblas_sdot(static_cast<int>(dimension), vec1, 1, vec1, 1);
    float dot_yy = cblas_sdot(static_cast<int>(dimension), vec2, 1, vec2, 1);
    float dot_xy = cblas_sdot(static_cast<int>(dimension), vec1, 1, vec2, 1);

    return dot_xx + dot_yy - 2.0f * dot_xy;
}
 