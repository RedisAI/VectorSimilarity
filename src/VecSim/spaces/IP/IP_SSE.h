#pragma once

float F_InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float F_InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float F_InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                          const void *qty_ptr);
float F_InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                           const void *qty_ptr);
