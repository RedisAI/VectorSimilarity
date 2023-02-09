/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <cstdlib>

void FP32_to_BF16_AVX512_SIMD16(const void *pVect1v, void *pVect2v, size_t qty);
