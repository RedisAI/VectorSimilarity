#pragma once

float L2SqrSIMD16Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float L2SqrSIMD16ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float L2SqrSIMD4Ext_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);
float L2SqrSIMD4ExtResiduals_SSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr);