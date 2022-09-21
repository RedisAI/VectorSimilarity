#pragma once

#include <cstdlib>

float FP32_L2SqrSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);

double FP64_L2SqrSIMD8Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD8ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
