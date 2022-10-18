#pragma once
#include <cstdlib>

double FP64_InnerProductSIMD2Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_InnerProductSIMD2ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                                 size_t qty);
