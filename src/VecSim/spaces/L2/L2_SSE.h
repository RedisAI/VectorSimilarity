#pragma once

#include <cstdlib>

float L2SqrSIMDsplit512Ext_SSE(const float *pVect1v, const float *pVect2v, size_t qty);
float L2SqrSIMD16ExtResiduals_SSE(const float *pVect1v, const float *pVect2v, size_t qty);
float L2SqrSIMD4Ext_SSE(const float *pVect1v, const float *pVect2v, size_t qty);
float L2SqrSIMD4ExtResiduals_SSE(const float *pVect1v, const float *pVect2v, size_t qty);

double L2SqrSIMDsplit512Ext_SSE(const double *pVect1v, const double *pVect2v, size_t qty);
double L2SqrSIMD8ExtResiduals_SSE(const double *pVect1v, const double *pVect2v, size_t qty);
double L2SqrSIMD2Ext_SSE(const double *pVect1v, const double *pVect2v, size_t qty);
double L2SqrSIMD2ExtResiduals_SSE(const double *pVect1v, const double *pVect2v, size_t qty);
