/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib>

float FP32_L2SqrSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);

double FP64_L2SqrSIMD8Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD8ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
