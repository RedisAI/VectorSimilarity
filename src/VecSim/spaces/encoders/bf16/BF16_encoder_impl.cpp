/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/encoders/bf16/BF16_encoder_impl.h"
#include "VecSim/vec_sim_common.h"
#include <memory.h>

void FP32_to_BF16_BigEndian(const void *pVect1v, void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    bf16 *pVect2 = (bf16 *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    while (pVect1 < pEnd1) {
        bf16 *orig = (bf16 *)(((char *)pVect1));
        *pVect2 = *orig;
        pVect1++;
        pVect2++;
    }
}

void FP32_to_BF16_LittleEndian(const void *pVect1v, void *pVect2v, size_t qty) {
    float *pVect1 = (float *)pVect1v;
    bf16 *pVect2 = (bf16 *)pVect2v;

    const float *pEnd1 = pVect1 + qty;

    while (pVect1 < pEnd1) {
        bf16 *orig = (bf16 *)(((char *)pVect1) + 2);
        *pVect2 = *orig;
        pVect1++;
        pVect2++;
    }
}

void BF16_to_FP32_BigEndian(const void *pVect1v, void *pVect2v, size_t qty) {
    bf16 *pVect1 = (bf16 *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect2 + qty;

    while (pVect2 < pEnd1) {
        bf16 *orig = (bf16 *)(((char *)pVect2));
        *orig = *pVect1;
        pVect1++;
        pVect2++;
    }
}

void BF16_to_FP32_LittleEndian(const void *pVect1v, void *pVect2v, size_t qty) {
    bf16 *pVect1 = (bf16 *)pVect1v;
    float *pVect2 = (float *)pVect2v;

    const float *pEnd1 = pVect2 + qty;

    while (pVect2 < pEnd1) {
        bf16 *orig = (bf16 *)(((char *)pVect2) + 2);
        *orig = *pVect1;
        pVect1++;
        pVect2++;
    }
}
