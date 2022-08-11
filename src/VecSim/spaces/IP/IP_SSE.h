#pragma once

float f_InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float f_InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float f_InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                          const void *qty_ptr);
float f_InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                           const void *qty_ptr);

double d_InnerProductSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
double d_InnerProductSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
double d_InnerProductSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                           const void *qty_ptr);
double d_InnerProductSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v,
                                            const void *qty_ptr);
