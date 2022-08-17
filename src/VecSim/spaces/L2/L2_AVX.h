#pragma once

#include <cstdlib>

float FP32_L2SqrSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
