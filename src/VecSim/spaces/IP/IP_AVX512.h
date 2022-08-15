#pragma once

float F_InnerProductSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float F_InnerProductSIMD4Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float F_InnerProductSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                              const void *qty_ptr);
float F_InnerProductSIMD4ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                             const void *qty_ptr);
