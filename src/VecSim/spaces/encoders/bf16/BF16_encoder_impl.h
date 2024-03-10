/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <cstdlib>

void FP32_to_BF16_BigEndian(const void *pVect1v, void *pVect2v, size_t qty);
void FP32_to_BF16_LittleEndian(const void *pVect1v, void *pVect2v, size_t qty);

void BF16_to_FP32_BigEndian(const void *pVect1v, void *pVect2v, size_t qty);
void BF16_to_FP32_LittleEndian(const void *pVect1v, void *pVect2v, size_t qty);
