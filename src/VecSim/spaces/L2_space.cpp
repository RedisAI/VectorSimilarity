

#include <cstdlib>
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_SSE.h"
namespace Spaces {

dist_func_ptr_ty<float> L2_FLOAT_GetOptDistFunc(size_t dim) {

    dist_func_ptr_ty<float> ret_dist_func = f_L2Sqr;
#if defined(M1)
#elif defined(__x86_64__)

    OptimizationScore optimization_type = GetDimOptimizationScore(dim);

    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__

        static dist_func_ptr_ty<float> dist_funcs[OPTIMIZATIONS_COUNT] = {
            f_L2SqrSIMD16Ext_AVX512, f_L2SqrSIMD4Ext_AVX512, f_L2SqrSIMD16ExtResiduals_AVX512,
            f_L2SqrSIMD4ExtResiduals_AVX512, f_L2Sqr};

        ret_dist_func = dist_funcs[optimization_type];
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__

        static dist_func_ptr_ty<float> dist_funcs[OPTIMIZATIONS_COUNT] = {
            f_L2SqrSIMD16Ext_AVX, f_L2SqrSIMD4Ext_AVX, f_L2SqrSIMD16ExtResiduals_AVX,
            f_L2SqrSIMD4ExtResiduals_AVX, f_L2Sqr};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    } else if (arch_opt == ARCH_OPT_SSE) {
#ifdef __SSE__

        static dist_func_ptr_ty<float> dist_funcs[OPTIMIZATIONS_COUNT] = {
            f_L2SqrSIMD16Ext_SSE, f_L2SqrSIMD4Ext_SSE, f_L2SqrSIMD16ExtResiduals_SSE,
            f_L2SqrSIMD4ExtResiduals_SSE, f_L2Sqr};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    }
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_ptr_ty<double> L2_DOUBLE_GetOptDistFunc(size_t dim) {

    dist_func_ptr_ty<double> ret_dist_func = d_L2Sqr;
#if defined(M1)
#elif defined(__x86_64__)

    OptimizationScore optimization_type = GetDimOptimizationScore(dim);

    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__

        static dist_func_ptr_ty<double> dist_funcs[OPTIMIZATIONS_COUNT] = {
            d_L2SqrSIMD16Ext_AVX512, d_L2SqrSIMD4Ext_AVX512, d_L2SqrSIMD16ExtResiduals_AVX512,
            d_L2SqrSIMD4ExtResiduals_AVX512, d_L2Sqr};

        ret_dist_func = dist_funcs[optimization_type];
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__

        static dist_func_ptr_ty<double> dist_funcs[OPTIMIZATIONS_COUNT] = {
            d_L2SqrSIMD16Ext_AVX, d_L2SqrSIMD4Ext_AVX, d_L2SqrSIMD16ExtResiduals_AVX,
            d_L2SqrSIMD4ExtResiduals_AVX, d_L2Sqr};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    } else if (arch_opt == ARCH_OPT_SSE) {
#ifdef __SSE__

        static dist_func_ptr_ty<double> dist_funcs[OPTIMIZATIONS_COUNT] = {
            d_L2SqrSIMD16Ext_SSE, d_L2SqrSIMD4Ext_SSE, d_L2SqrSIMD16ExtResiduals_SSE,
            d_L2SqrSIMD4ExtResiduals_SSE, d_L2Sqr};

        ret_dist_func = dist_funcs[optimization_type];

#endif
    }
#endif // __x86_64__
    return ret_dist_func;
}
} // namespace Spaces
