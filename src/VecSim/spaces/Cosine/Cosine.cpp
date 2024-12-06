/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "Cosine.h"

float INT8_Cosine(const void *pVect1v, const void *pVect2v, size_t dimension) {
    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;

    int res = 0;
    for (size_t i = 0; i < dimension; i++) {
        int16_t a = pVect1[i];
        int16_t b = pVect2[i];
        res += a * b;
    }

    float norm_v1 = *(float *)pVect1v;
    float norm_v2 = *(float *)pVect2v;
    return 1.0f - float(res) / (norm_v1 * norm_v2);
}
