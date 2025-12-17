/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cstdlib>

// pVect1v vector of type fp32 and pVect2v vector of type uint8
float SQ8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

float FP32_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

double FP64_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

float BF16_L2Sqr_LittleEndian(const void *pVect1v, const void *pVect2v, size_t dimension);
float BF16_L2Sqr_BigEndian(const void *pVect1v, const void *pVect2v, size_t dimension);

float FP16_L2Sqr(const void *pVect1, const void *pVect2, size_t dimension);

float INT8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

float UINT8_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);

// 4-bit packed vectors: two 4-bit values per byte
float INT4_L2Sqr(const void *pVect1v, const void *pVect2v, size_t dimension);
