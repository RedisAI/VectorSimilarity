/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include <armpl.h>

float FP32_L2Sqr_ARMPL_NEON(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const float *vec1 = static_cast<const float *>(pVect1v);
    const float *vec2 = static_cast<const float *>(pVect2v);

    float result = 0.0f;
    constexpr const size_t blockSize = 1024;
    float buffer[blockSize];

    // Pre-calculate number of full blocks and the size of the last partial block
    const size_t fullBlockCount = dimension / blockSize;
    const size_t lastBlockSize = dimension % blockSize;

    // Process full blocks
    for (size_t i = 0; i < fullBlockCount; i++) {
        size_t offset = i * blockSize;

        // Calculate difference vector for full block
        for (size_t j = 0; j < blockSize; j++) {
            buffer[j] = vec1[offset + j] - vec2[offset + j];
        }

        // Use ARMPL to compute dot product
        result += cblas_sdot(blockSize, buffer, 1, buffer, 1);
    }

    // Handle remaining elements (if any)
    if (lastBlockSize > 0) {
        size_t offset = fullBlockCount * blockSize;

        // Calculate difference vector for remaining elements
        for (size_t j = 0; j < lastBlockSize; j++) {
            buffer[j] = vec1[offset + j] - vec2[offset + j];
        }

        // Use ARMPL to compute dot product
        result += cblas_sdot(lastBlockSize, buffer, 1, buffer, 1);
    }

    return result;
}
