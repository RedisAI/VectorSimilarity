/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/space_includes.h"
#include "armpl.h"

template <unsigned char residual>
double FP64_L2Sqr_ARMPL_SVE2(const void *pVect1v, const void *pVect2v, size_t dimension) {
    const double *vec1 = static_cast<const double *>(pVect1v);
    const double *vec2 = static_cast<const double *>(pVect2v);

    double result = 0.0f;
    const size_t blockSize = 512; // Changed from int to size_t
    double buffer[blockSize];

    for (size_t i = 0; i < dimension; i += blockSize) {
        // Process in smaller chunks to improve cache behavior
        size_t currentBlock = std::min(blockSize, dimension - i);

        // Calculate difference vector in chunks
        for (size_t j = 0; j < currentBlock; j++) {
            buffer[j] = vec1[i + j] - vec2[i + j];
        }

        // Use ArmPL to compute dot product of difference with itself
        result += cblas_ddot(currentBlock, buffer, 1, buffer, 1);
    }

    return result;
}