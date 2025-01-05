/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib>

float FP32_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

double FP64_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

float BF16_L2Sqr_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension);
float BF16_L2Sqr_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension);

float FP16_L2Sqr(const void *pVect1, const void *pVect2, size_t dimension);

float INT8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

float UINT8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);
