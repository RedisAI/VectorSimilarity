#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/L2/L2.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/L2/L2_AVX512.h"
#include "VecSim/spaces/L2/L2_AVX.h"
#include "VecSim/spaces/L2/L2_SSE.h"
namespace spaces {

template <typename DistType>
dist_func_t<DistType> Return_SSE(CalculationGuideline optimization_type) {
    static dist_func_t<DistType> dist_funcs[] = {L2Sqr, L2SqrSIMDsplit512Ext_SSE, L2SqrSIMD4Ext_SSE,
                                                 L2SqrSIMD16ExtResiduals_SSE,
                                                 L2SqrSIMD4ExtResiduals_SSE};

    return dist_funcs[optimization_type];
}

dist_func_t<float> L2_FP32_GetDistFunc(size_t dim) {

    dist_func_t<float> ret_dist_func = L2Sqr;
#if defined(M1)
#elif defined(__x86_64__)

    CalculationGuideline optimization_type = FP32_GetCalculationGuideline(dim);
    switch (arch_opt) {
    case ARCH_OPT_NONE:
        break;
    case ARCH_OPT_AVX512:
#ifdef __AVX512F__
    {
        static dist_func_t<float> dist_funcs[] = {
            L2Sqr, FP32_L2SqrSIMD16Ext_AVX512, FP32_L2SqrSIMD4Ext_AVX512,
            FP32_L2SqrSIMD16ExtResiduals_AVX512, FP32_L2SqrSIMD4ExtResiduals_AVX512};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    case ARCH_OPT_AVX:
#ifdef __AVX__
    {
        static dist_func_t<float> dist_funcs[] = {
            L2Sqr, FP32_L2SqrSIMD16Ext_AVX, FP32_L2SqrSIMD4Ext_AVX,
            FP32_L2SqrSIMD16ExtResiduals_AVX, FP32_L2SqrSIMD4ExtResiduals_AVX};

        ret_dist_func = dist_funcs[optimization_type];
    } break;

#endif
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        ret_dist_func = Return_SSE<float>(optimization_type);
    } break;
    } // switch
#endif

#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> L2_FP64_GetDistFunc(size_t dim) {

    dist_func_t<double> ret_dist_func = FP64_L2Sqr;
#if defined(M1)
#elif defined(__x86_64__)

    CalculationGuideline optimization_type = FP64_GetCalculationGuideline(dim);
    switch (arch_opt) {
    case ARCH_OPT_NONE:
        break;
    case ARCH_OPT_AVX512:
    case ARCH_OPT_AVX:
    case ARCH_OPT_SSE:
#ifdef __SSE__
    {
        static dist_func_t<double> dist_funcs[] = {
            FP64_L2Sqr, FP64_L2SqrSIMD8Ext_SSE, FP64_L2SqrSIMD2Ext_SSE,
            FP64_L2SqrSIMD8ExtResiduals_SSE, FP64_L2SqrSIMD2ExtResiduals_SSE};

        ret_dist_func = dist_funcs[optimization_type];
    } break;
#endif
    } // switch

    #endif // __x86_64__ */
    return ret_dist_func;
}

} // namespace spaces
