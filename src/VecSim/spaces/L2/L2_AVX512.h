#pragma once

#include <cstdlib>

float FP32_L2SqrSIMD16Ext_AVX512(const float *pVect1v, const float *pVect2v, size_t qty);
float FP32_L2SqrSIMD16ExtResiduals_AVX512(const float *pVect1v, const float *pVect2v, size_t qty);
float FP32_L2SqrSIMD4Ext_AVX512(const float *pVect1v, const float *pVect2v, size_t qty);
float FP32_L2SqrSIMD4ExtResiduals_AVX512(const float *pVect1v, const float *pVect2v, size_t qty);
