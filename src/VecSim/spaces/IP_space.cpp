#include "IP_space.h"
#include "space_aux.h"
#include "VecSim/spaces/IP/IP.h"

InnerProductSpace::InnerProductSpace(size_t dim, std::shared_ptr<VecSimAllocator> allocator)
    : SpaceInterface(allocator) {
    fstdistfunc_ = InnerProduct;
#if defined(__x86_64__)
    Arch_Optimization arch_opt = getArchitectureOptimization();
    if (arch_opt == ARCH_OPT_AVX512) {
#include "VecSim/spaces/IP/IP_AVX512.h"
#ifdef __AVX512F__
        if (dim % 16 == 0) {
            fstdistfunc_ = InnerProductSIMD16Ext_AVX512;
        } else {
            fstdistfunc_ = InnerProductSIMD16ExtResiduals_AVX512;
        }
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__
#include "VecSim/spaces/IP/IP_AVX.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = InnerProductSIMD16Ext_AVX;
        } else if (dim % 4 == 0) {
            fstdistfunc_ = InnerProductSIMD4Ext_AVX;
        } else if (dim > 16) {
            fstdistfunc_ = InnerProductSIMD16ExtResiduals_AVX;
        } else if (dim > 4) {
            fstdistfunc_ = InnerProductSIMD4ExtResiduals_AVX;
        }
#endif
    } else if (arch_opt == ARCH_OPT_SSE) {
#ifdef __SSE__
#include "VecSim/spaces/IP/IP_SSE.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = InnerProductSIMD16Ext_SSE;
        } else if (dim % 4 == 0) {
            fstdistfunc_ = InnerProductSIMD4Ext_SSE;
        } else if (dim > 16) {
            fstdistfunc_ = InnerProductSIMD16ExtResiduals_SSE;
        } else if (dim > 4) {
            fstdistfunc_ = InnerProductSIMD4ExtResiduals_SSE;
        }
#endif
    }
#endif // __x86_64__
    dim_ = dim;
    data_size_ = dim * sizeof(float);
}

size_t InnerProductSpace::get_data_size() const { return data_size_; }

DISTFUNC<float> InnerProductSpace::get_dist_func() const { return fstdistfunc_; }

void *InnerProductSpace::get_data_dim() { return &dim_; }

InnerProductSpace::~InnerProductSpace() = default;
