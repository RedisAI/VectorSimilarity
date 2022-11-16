/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <cstdlib>
float FP32_InnerProductSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);

double FP64_InnerProductSIMD8Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD8ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD2Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD2ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
