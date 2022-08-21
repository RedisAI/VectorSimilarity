#pragma once

float FP32_InnerProductSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v,
                                        const void *qty_ptr);
float FP32_InnerProductSIMD4Ext_AVX512(const void *pVect1v, const void *pVect2v,
                                       const void *qty_ptr);
float FP32_InnerProductSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                                 const void *qty_ptr);
float FP32_InnerProductSIMD4ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                                const void *qty_ptr);
