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
 #include "VecSim/spaces/functions/NEON.h"
 #include "VecSim/spaces/functions/SVE.h"
 #include "VecSim/spaces/functions/SVE2.h"
 
 namespace spaces {
 dist_func_t<float> IP_FP32_GetDistFunc(size_t dim, const Arch_Optimization arch_opt,
                                        unsigned char *alignment) {
     unsigned char dummy_alignment;
     if (alignment == nullptr) {
         alignment = &dummy_alignment;
     }
 
     dist_func_t<float> ret_dist_func = FP32_InnerProduct;
 
#ifdef CPU_FEATURES_ARCH_X86_64
    CalculationGuideline optimization_type = FP32_GetCalculationGuideline(dim);
     // Optimizations assume at least 16 floats. If we have less, we use the naive implementation.
     if (dim < 16) {
         return ret_dist_func;
     }
#endif
     switch (arch_opt) {
 #ifdef CPU_FEATURES_ARCH_X86_64
    case ARCH_OPT_AVX512_DQ:
     case ARCH_OPT_AVX512_F:

 #ifdef OPT_AVX512F
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_AVX512, FP32_InnerProductSIMD4Ext_AVX512,
            FP32_InnerProductSIMD16ExtResiduals_AVX512, FP32_InnerProductSIMD4ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
 #endif
     case ARCH_OPT_AVX:
 #ifdef OPT_AVX
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_AVX, FP32_InnerProductSIMD4Ext_AVX,
            FP32_InnerProductSIMD16ExtResiduals_AVX, FP32_InnerProductSIMD4ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

 #endif
     case ARCH_OPT_SSE:
 #ifdef OPT_SSE
    {
        static dist_func_t<float> dist_funcs[] = {
            FP32_InnerProduct, FP32_InnerProductSIMD16Ext_SSE, FP32_InnerProductSIMD4Ext_SSE,
            FP32_InnerProductSIMD16ExtResiduals_SSE, FP32_InnerProductSIMD4ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
 #endif
 #endif // __x86_64__
 #ifdef CPU_FEATURES_ARCH_AARCH64
     case ARCH_OPT_SVE2:
 #ifdef OPT_SVE2
         ret_dist_func = Choose_FP32_IP_implementation_SVE2(dim);
         break;
 
 #endif
     case ARCH_OPT_SVE:
 #ifdef OPT_SVE
         ret_dist_func = Choose_FP32_IP_implementation_SVE(dim);
         break;
 
 #endif
     case ARCH_OPT_NEON:
 #ifdef OPT_NEON
         ret_dist_func = Choose_FP32_IP_implementation_NEON(dim);
         break;
 #endif
 #endif // CPU_FEATURES_ARCH_AARCH64
     case ARCH_OPT_NONE:
         break;
     } // switch
     return ret_dist_func;
 }
 
dist_func_t<double> IP_FP64_GetDistFunc(size_t dim, const Arch_Optimization arch_opt) {
 
     dist_func_t<double> ret_dist_func = FP64_InnerProduct;
#ifdef CPU_FEATURES_ARCH_X86_64
 
    CalculationGuideline optimization_type = FP64_GetCalculationGuideline(dim);
#endif

    switch (arch_opt) {
#ifdef CPU_FEATURES_ARCH_X86_64
    case ARCH_OPT_AVX512_DQ:
#ifdef OPT_AVX512DQ
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX512, FP64_InnerProductSIMD2Ext_AVX512,
            FP64_InnerProductSIMD8ExtResiduals_AVX512, FP64_InnerProductSIMD2ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
     case ARCH_OPT_AVX512_F:
 #ifdef OPT_AVX512F
    {
        // If AVX512 foundation flag is supported, but AVX512DQ isn't supported, we cannot extract
        // 2X64-bit elements from the 512bit register, which is required when dim%8 != 0, so we can
        // continue the vector computations by using 128 register optimization on the vectors'
        // tails. Then, we use modified versions that split both part of the computation without
        // using the unsupported extraction operation.
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX512,
            FP64_InnerProductSIMD8ExtResiduals_AVX512, FP64_InnerProductSIMD2Ext_AVX512_noDQ,
            FP64_InnerProductSIMD2ExtResiduals_AVX512_noDQ};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
 #endif
     case ARCH_OPT_AVX:
 #ifdef OPT_AVX
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_AVX, FP64_InnerProductSIMD2Ext_AVX,
            FP64_InnerProductSIMD8ExtResiduals_AVX, FP64_InnerProductSIMD2ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

 #endif
     case ARCH_OPT_SSE:
 #ifdef OPT_SSE
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_InnerProduct, FP64_InnerProductSIMD8Ext_SSE, FP64_InnerProductSIMD2Ext_SSE,
            FP64_InnerProductSIMD8ExtResiduals_SSE, FP64_InnerProductSIMD2ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
 #endif
 #endif // __x86_64__ */
 #ifdef CPU_FEATURES_ARCH_AARCH64
     case ARCH_OPT_SVE2:
 #ifdef OPT_SVE2
         ret_dist_func = Choose_FP64_IP_implementation_SVE2(dim);
         break;
 
 #endif
     case ARCH_OPT_SVE:
 #ifdef OPT_SVE
         ret_dist_func = Choose_FP64_IP_implementation_SVE(dim);
         break;
 
 #endif
     case ARCH_OPT_NEON:
 #ifdef OPT_NEON
         ret_dist_func = Choose_FP64_IP_implementation_NEON(dim);
         break;
 #endif
 #endif // CPU_FEATURES_ARCH_AARCH64
     case ARCH_OPT_NONE:
         break;
     } // switch
     return ret_dist_func;
 }
 
 } // namespace spaces
 