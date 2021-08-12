#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

namespace hnswlib {
template <typename TYPE> using DISTFUNC = TYPE (*)(const void *, const void *, const void *);

template <typename TYPE> class SpaceInterface {
  public:
    virtual size_t get_data_size() const = 0;

    virtual DISTFUNC<TYPE> get_dist_func() const = 0;

    virtual void *get_data_dim() = 0;

    virtual ~SpaceInterface() = default;
};
} // namespace hnswlib
