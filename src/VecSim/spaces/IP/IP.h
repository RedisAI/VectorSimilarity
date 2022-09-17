#pragma once

#include <cstdlib>

float InnerProduct(const float *pVect1, const float *pVect2, size_t qty);

float InnerProduct_impl(const float *pVect1, const float *pVect2, size_t qty);

double InnerProduct(const double *pVect1, const double *pVect2, size_t qty);

double InnerProduct_impl(const double *pVect1, const double *pVect2, size_t qty);
