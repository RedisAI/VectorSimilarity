#pragma once

float InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                        const void *qty_ptr);
float InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                         const void *qty_ptr);
