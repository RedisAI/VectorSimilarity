/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "armpl.h"

float FP32_L2Sqr_ARMPL_SVE(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *vec1 = static_cast<const float *>(pVect1v);
    const float *vec2 = static_cast<const float *>(pVect2v);

    float result = 0.0f;
    constexpr const size_t blockSize = 1024;
    float buffer[blockSize];

    for (size_t i = 0; i < dimension; i += blockSize) {
        // Process in smaller chunks to improve cache behavior
        size_t currentBlock = std::min(blockSize, dimension - i);

        // Calculate difference vector in chunks
        for (size_t j = 0; j < currentBlock; j++) {
            buffer[j] = vec1[i + j] - vec2[i + j];
        }

        // Notice: Armpl can choose different implementation based on cpu features.
        result += cblas_sdot(currentBlock, buffer, 1, buffer, 1);
    }

    return result;
}
