#pragma once

float f_InnerProductSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float f_InnerProductSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float f_InnerProductSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v,
                                           const void *qty_ptr);
float f_InnerProductSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v,
                                          const void *qty_ptr);
