#pragma once
#include <cstdlib>

double FP64_L2SqrSIMD2Ext_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
double FP64_L2SqrSIMD2ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v, size_t qty);
