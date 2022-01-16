#pragma once

#ifdef __cplusplus
extern "C" {
#endif

float InnerProductSIMD16Ext_AVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float InnerProductSIMD16ExtResiduals_AVX512(const void *pVect1v, const void *pVect2v,
                                            const void *qty_ptr);

#ifdef __cplusplus
}
#endif
