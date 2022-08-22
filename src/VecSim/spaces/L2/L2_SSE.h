#pragma once

#include <cstdlib>

float FP32_L2SqrSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
