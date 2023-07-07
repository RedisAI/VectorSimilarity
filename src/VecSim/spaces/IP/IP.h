/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib>

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension);

double FP64_InnerProduct(const void *pVect1, const void *pVect2, size_t dimension);
