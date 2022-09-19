#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/IP/IP.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/IP/IP_AVX512.h"
#include "VecSim/spaces/IP/IP_AVX.h"
#include "VecSim/spaces/IP/IP_SSE.h"

namespace spaces {
dist_func_t<float> IP_FP32_GetDistFunc(size_t dim) {

    dist_func_t<float> ret_dist_func = FP32_InnerProduct;
#if defined(M1)
#elif defined(__x86_64__)

    CalculationGuideline optimization_type = GetCalculationGuideline(dim);
    switch (arch_opt) {
    case ARCH_OPT_NONE:
        break;
    case ARCH_OPT_AVX512:
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
    } // switch
#endif // __x86_64__
    return ret_dist_func;
}

dist_func_t<double> IP_FP64_GetDistFunc(size_t dim) { return FP64_InnerProduct; }

} // namespace spaces
