#pragma once

#include <cstdlib>

float FP32_L2SqrSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_L2SqrSIMD4ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);

double FP64_L2SqrSIMD8Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD8ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
