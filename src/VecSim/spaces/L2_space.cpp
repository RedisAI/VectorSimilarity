

#include <cstdlib>
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_SSE.h"
namespace Spaces {

dist_func_t<float> L2_FLOAT_GetOptDistFunc(size_t dim) {

    dist_func_t<float> ret_dist_func = F_L2Sqr;
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
            F_L2Sqr, F_L2SqrSIMD16Ext_AVX512, F_L2SqrSIMD4Ext_AVX512,
            F_L2SqrSIMD16ExtResiduals_AVX512, F_L2SqrSIMD4ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    }
        break;
#endif
    case ARCH_OPT_AVX: 
#ifdef __AVX__
        {
        static dist_func_t<float> dist_funcs[] = {
            F_L2Sqr, F_L2SqrSIMD16Ext_AVX, F_L2SqrSIMD4Ext_AVX, F_L2SqrSIMD16ExtResiduals_AVX,
            F_L2SqrSIMD4ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
        }
        break;

#endif
    case ARCH_OPT_SSE: 
#ifdef __SSE__
        {
        static dist_func_t<float> dist_funcs[] = {
            F_L2Sqr, F_L2SqrSIMD16Ext_SSE, F_L2SqrSIMD4Ext_SSE, F_L2SqrSIMD16ExtResiduals_SSE,
            F_L2SqrSIMD4ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    }
        break;
    } // switch
#endif

#endif // __x86_64__
    return ret_dist_func;
}

} // namespace Spaces
