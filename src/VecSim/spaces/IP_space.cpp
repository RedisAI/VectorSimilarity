/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/IP/IP_AVX512DQ.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"

namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<float> ret_dist_func = FP32_InnerProduct;
#if defined(M1)
#elif defined(__x86_64__)

    CalculationGuideline optimization_type = FP32_GetCalculationGuideline(dim);

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_AVX512, FP32_InnerProductSIMD4Ext_AVX512,
            FP32_InnerProductSIMD16ExtResiduals_AVX512, FP32_InnerProductSIMD4ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_AVX, FP32_InnerProductSIMD4Ext_AVX,
            FP32_InnerProductSIMD16ExtResiduals_AVX, FP32_InnerProductSIMD4ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_SSE, FP32_InnerProductSIMD4Ext_SSE,
            FP32_InnerProductSIMD16ExtResiduals_SSE, FP32_InnerProductSIMD4ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {

    dist_func_t<double> ret_dist_func = FP64_InnerProduct;
#if defined(M1)
#elif defined(__x86_64__)

    CalculationGuideline optimization_type = FP64_GetCalculationGuideline(dim);

    switch (arch_opt) {
    case ARCH_OPT_AVX512_DQ:
#ifdef __AVX512DQ__
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX512, FP64_InnerProductSIMD2Ext_AVX512,
            FP64_InnerProductSIMD8ExtResiduals_AVX512, FP64_InnerProductSIMD2ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX512_F:
#ifdef __AVX512F__
    {
        // If AVX512 foundation flag is supported, but AVX512DQ isn't supported, we cannot extract
        // 2X64-bit elements from the 512bit register, which is required when dim%8 != 0, so we can
        // continue the vector computations by using 128 register optimization on the vectors'
        // tails. Then, we use modified versions that split both part of the computation without
        // using the unsupported extraction operation.
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX512,
            FP64_InnerProductSIMD2Ext_AVX512_noDQ, FP64_InnerProductSIMD8ExtResiduals_AVX512,
            FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX, FP64_InnerProductSIMD2Ext_AVX,
            FP64_InnerProductSIMD8ExtResiduals_AVX, FP64_InnerProductSIMD2ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_SSE, FP64_InnerProductSIMD2Ext_SSE,
            FP64_InnerProductSIMD8ExtResiduals_SSE, FP64_InnerProductSIMD2ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_NONE:
        break;
    } // switch
#endif // __x86_64__ */
    return ret_dist_func;
}

} // namespace spaces
