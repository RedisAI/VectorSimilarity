#pragma once

#include <cstdlib>
float FP32_InnerProductSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
float FP32_InnerProductSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v, size_t qty);
