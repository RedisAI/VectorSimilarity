/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/conveters/bf16/BF16_converter_impl.h"
#include "VecSim/vec_sim_common.h"
#include <memory.h>

void FP32_to_BF16(const void *pVect1v, void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    bf16 *pVect2 = (bf16 *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    while (pVect1 < pEnd1) {

        memcpy(pVect2, pVect1, sizeof(bf16));
        pVect1++;
        pVect2++;
    }
}

