#pragma once
#include <cstdlib>

float FP32_InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);

double FP64_InnerProductSIMD8Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD2Ext_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD2ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD8Ext Residuals_SSE(const void *pVect1v, const void *pVect2v, size_t qty);
