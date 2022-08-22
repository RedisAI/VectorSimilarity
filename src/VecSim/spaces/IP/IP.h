#pragma once

#include <cstdlib>

float FP32_InnerProduct(const void *pVect1, const void *pVect2, size_t qty);

float FP32_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty);
