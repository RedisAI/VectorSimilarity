
#include <cstdlib>
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"
namespace Spaces {

dist_func_t<float> IP_FLOAT_GetOptDistFunc(size_t dim) {

    dist_func_t<float> ret_dist_func = F_InnerProduct;
#if defined(M1)
#elif defined(__x86_64__)

    OptimizationScore optimization_type = GetDimOptimizationScore(dim);
    switch (arch_opt) {
    case ARCH_OPT_NONE:
        break;
    case ARCH_OPT_AVX512:
#ifdef __AVX512F__
    {
        static dist_func_t<float> dist_funcs[] = {
            F_InnerProduct, F_InnerProductSIMD16Ext_AVX512, F_InnerProductSIMD4Ext_AVX512,
            F_InnerProductSIMD16ExtResiduals_AVX512, F_InnerProductSIMD4ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
    {
        static dist_func_t<float> dist_funcs[] = {
            F_InnerProduct, F_InnerProductSIMD16Ext_AVX, F_InnerProductSIMD4Ext_AVX,
            F_InnerProductSIMD16ExtResiduals_AVX, F_InnerProductSIMD4ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        static dist_func_t<float> dist_funcs[] = {
            F_InnerProduct, F_InnerProductSIMD16Ext_SSE, F_InnerProductSIMD4Ext_SSE,
            F_InnerProductSIMD16ExtResiduals_SSE, F_InnerProductSIMD4ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    } // switch
#endif // __x86_64__
    return ret_dist_func;
}

} // namespace Spaces
