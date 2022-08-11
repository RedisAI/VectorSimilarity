
#include <cstdlib>
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"
namespace Spaces {

dist_func_ptr_ty<float> IP_FLOAT_GetDistFunc(size_t dim) {

#if defined(M1)
#elif defined(__x86_64__)

    dist_func_ptr_ty<float> ret_dist_func;
    OptimizationScore optimization_type = GetDimOptimizationScore(dim);

    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__

        static dist_func_ptr_ty<float> dist_funcs[OPTIMIZATIONS_COUNT] = {
            f_InnerProductSIMD16Ext_AVX512, f_InnerProductSIMD4Ext_AVX512,
            f_InnerProductSIMD16ExtResiduals_AVX512, f_InnerProductSIMD4ExtResiduals_AVX512,
            f_InnerProduct};

        ret_dist_func = dist_funcs[optimization_type];
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__

        static dist_func_ptr_ty<float> dist_funcs[OPTIMIZATIONS_COUNT] = {
            f_InnerProductSIMD16Ext_AVX, f_InnerProductSIMD4Ext_AVX,
            f_InnerProductSIMD16ExtResiduals_AVX, f_InnerProductSIMD4ExtResiduals_AVX,
            f_InnerProduct};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    } else if (arch_opt == ARCH_OPT_SSE) {
#ifdef __SSE__

        static dist_func_ptr_ty<float> dist_funcs[OPTIMIZATIONS_COUNT] = {
            f_InnerProductSIMD16Ext_SSE, f_InnerProductSIMD4Ext_SSE,
            f_InnerProductSIMD16ExtResiduals_SSE, f_InnerProductSIMD4ExtResiduals_SSE,
            f_InnerProduct};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    }
#endif // __x86_64__

    return ret_dist_func;
}

dist_func_ptr_ty<double> IP_DOUBLE_GetDistFunc(size_t dim) {

#if defined(M1)
#elif defined(__x86_64__)

    dist_func_ptr_ty<double> ret_dist_func;
    OptimizationScore optimization_type = GetDimOptimizationScore(dim);

    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__

        static dist_func_ptr_ty<double> dist_funcs[OPTIMIZATIONS_COUNT] = {
            d_InnerProductSIMD16Ext_AVX512, d_InnerProductSIMD4Ext_AVX512,
            d_InnerProductSIMD16ExtResiduals_AVX512, d_InnerProductSIMD4ExtResiduals_AVX512,
            d_InnerProduct};

        ret_dist_func = dist_funcs[optimization_type];
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__

        static dist_func_ptr_ty<double> dist_funcs[OPTIMIZATIONS_COUNT] = {
            d_InnerProductSIMD16Ext_AVX, d_InnerProductSIMD4Ext_AVX,
            d_InnerProductSIMD16ExtResiduals_AVX, d_InnerProductSIMD4ExtResiduals_AVX,
            d_InnerProduct};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    } else if (arch_opt == ARCH_OPT_SSE) {
#ifdef __SSE__

        static dist_func_ptr_ty<double> dist_funcs[OPTIMIZATIONS_COUNT] = {
            d_InnerProductSIMD16Ext_SSE, d_InnerProductSIMD4Ext_SSE,
            d_InnerProductSIMD16ExtResiduals_SSE, d_InnerProductSIMD4ExtResiduals_SSE,
            d_InnerProduct};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    }
#endif // __x86_64__

    return ret_dist_func;
}
} // namespace Spaces
