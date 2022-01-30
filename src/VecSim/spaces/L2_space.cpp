#include "L2_space.h"
#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/L2/L2.h"

#include <cstdlib>

// clang-format off

L2Space::L2Space(size_t dim, std::shared_ptr<VecSimAllocator> allocator)
    : SpaceInterface(allocator) {
    fstdistfunc_ = L2Sqr;
#if defined(__x86_64__)
    Arch_Optimization arch_opt = getArchitectureOptimization();

#if defined(__AVX512F__)
    if (arch_opt == ARCH_OPT_AVX512) {
#ifdef __AVX512F__
#include "VecSim/spaces/L2/L2_AVX512.h"
        if (dim % 16 == 0) {
            fstdistfunc_ = L2SqrSIMD16Ext_AVX512;
        } else {
            fstdistfunc_ = L2SqrSIMD16ExtResiduals_AVX512;
        }
    } else

#elif defined(__AVX__)
    if (arch_opt == ARCH_OPT_AVX) {
        if (dim % 16 == 0) {
            fstdistfunc_ = L2SqrSIMD16Ext_AVX;
        } else {
            fstdistfunc_ = L2SqrSIMD16ExtResiduals_AVX;
        }
    } else

#elif defined(__SSE__)
    if (arch_opt == ARCH_OPT_SSE) {
        if (dim % 16 == 0) {
            fstdistfunc_ = L2SqrSIMD16Ext_SSE;
        } else if (dim % 4 == 0) {
            fstdistfunc_ = L2SqrSIMD4Ext_SSE;
        } else if (dim > 16) {
            fstdistfunc_ = L2SqrSIMD16ExtResiduals_SSE;
        } else if (dim > 4) {
            fstdistfunc_ = L2SqrSIMD4ExtResiduals_SSE;
        }
    } else
#endif

	{ (void) arch_opt; }
#endif // __x86_64__

    dim_ = dim;
    data_size_ = dim * sizeof(float);
}

// clang-format on

size_t L2Space::get_data_size() const { return data_size_; }

DISTFUNC<float> L2Space::get_dist_func() const { return fstdistfunc_; }

void *L2Space::get_data_dim() { return &dim_; }

L2Space::~L2Space() = default;
