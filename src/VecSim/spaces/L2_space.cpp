#include <cstdlib>
#include "L2_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/L2/L2.h"

L2Space::L2Space(size_t dim, std::shared_ptr<VecSimAllocator> allocator)
    : SpaceInterface(allocator) {
    fstdistfunc_ = L2Sqr;
#if defined(__x86_64__)
    Arch_Optimization arch_opt = getArchitectureOptimization();
    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__
#include "VecSim/spaces/L2/L2_AVX512.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = L2SqrSIMD16Ext_AVX512;
        } else {
            fstdistfunc_ = L2SqrSIMD16ExtResiduals_AVX512;
        }
#endif
    } else if (arch_opt == ARCH_OPT_AVX) {
#ifdef __AVX__
#include "VecSim/spaces/L2/L2_AVX.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = L2SqrSIMD16Ext_AVX;
        } else {
            fstdistfunc_ = L2SqrSIMD16ExtResiduals_AVX;
        }
#endif
    } else if (arch_opt == ARCH_OPT_SSE) {
#ifdef __SSE__
#include "VecSim/spaces/L2/L2_SSE.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = L2SqrSIMD16Ext_SSE;
        } else if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrSIMD4Ext_SSE;
        } else if (dim > 16) {
            fstdistfunc_ = L2SqrSIMD16ExtResiduals_SSE;
        } else if (dim > 4) {
            fstdistfunc_ = L2SqrSIMD4ExtResiduals_SSE;
        }
#endif
    }
#endif // __x86_64__
    dim_ = dim;
    data_size_ = dim * sizeof(float);
}

size_t L2Space::get_data_size() const { return data_size_; }

DISTFUNC<float> L2Space::get_dist_func() const { return fstdistfunc_; }

void *L2Space::get_data_dim() { return &dim_; }

L2Space::~L2Space() = default;
