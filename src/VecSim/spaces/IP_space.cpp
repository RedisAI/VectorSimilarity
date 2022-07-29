#include "IP_space.h"
#include "space_aux.h"
#include "VecSim/spaces/IP/IP.h"

InnerProductSpace::InnerProductSpace(size_t dim, std::shared_ptr<VecSimAllocator> allocator)
    : SpaceInterface(allocator) {
    fstdistfunc_ = InnerProduct;

    // General optimization logic:
    // SIMD16 perform computations on 16 float at a time in each iteration, while SIMD4 perform
    // computations on 16 float on most of the vector, and on the residual performing on 4 floats at
    // a time.
    // When we have a dimension that is not divisible by 4, we should use SIMD16ExtResiduals only if
    // the reminder is less than 4, because otherwise we can still perform SIMD4 operations.
#if defined(M1)

#elif defined(__x86_64__)
    Arch_Optimization arch_opt = getArchitectureOptimization();
    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__
#include "VecSim/spaces/IP/IP_AVX512.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = InnerProductSIMD16Ext_AVX512;
        } else if (dim % 4 == 0) {
            fstdistfunc_ = InnerProductSIMD4Ext_AVX512;
        } else if (dim > 16 && dim % 16 < 4) {
            fstdistfunc_ = InnerProductSIMD16ExtResiduals_AVX512;
        } else if (dim > 4) {
            fstdistfunc_ = InnerProductSIMD4ExtResiduals_AVX512;
        }
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__
#include "VecSim/spaces/IP/IP_AVX.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = InnerProductSIMD16Ext_AVX;
        } else if (dim % 4 == 0) {
            fstdistfunc_ = InnerProductSIMD4Ext_AVX;
        } else if (dim > 16 && dim % 16 < 4) {
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
        } else if (dim > 16 && dim % 16 < 4) {
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
