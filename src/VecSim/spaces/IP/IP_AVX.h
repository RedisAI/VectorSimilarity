#pragma once

float InnerProductSIMD16Ext_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProductSIMD4Ext_AVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProductSIMD16ExtResiduals_AVX(const void *pVect1v, const void *pVect2v,
                                         const void *qty_ptr);
float InnerProductSIMD4ExtResiduals_AVX(const void *pVect1v, const void *pVect2v,
                                        const void *qty_ptr);
