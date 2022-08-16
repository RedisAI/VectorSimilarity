#pragma once

float FP32_L2SqrSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float FP32_L2SqrSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                          const void *qty_ptr);
float FP32_L2SqrSIMD4Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float FP32_L2SqrSIMD4ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                         const void *qty_ptr);
