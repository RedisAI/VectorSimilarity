/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/vec_sim_common.h"
#include <stddef.h>
#include <memory>
#include <atomic>
#include <cstring>

struct VecSimAllocator {
    // Allow global vecsim memory functions to access this class.
    friend inline void *vecsim_malloc(size_t n);
    friend inline void *vecsim_calloc(size_t nelem, size_t elemsz);
    friend inline void *vecsim_realloc(void *p, size_t n);
    friend inline void vecsim_free(void *p);

private:
    std::atomic_uint64_t allocated;

    // Static member that indicates each allocation additional size.
    static size_t allocation_header_size;
    static VecSimMemoryFunctions memFunctions;

    VecSimAllocator() : allocated(std::atomic_uint64_t(sizeof(VecSimAllocator))) {}

public:
    static std::shared_ptr<VecSimAllocator> newVecsimAllocator();
    void *allocate(size_t size);
    void *callocate(size_t size);
    void deallocate(void *p, size_t size);
    void *reallocate(void *p, size_t size);
    void free_allocation(void *p);

    void *operator new(size_t size);
    void *operator new[](size_t size);
    void operator delete(void *p, size_t size);
    void operator delete[](void *p, size_t size);

    int64_t getAllocationSize() const;
    inline friend bool operator==(const VecSimAllocator &a, const VecSimAllocator &b) {
        return a.allocated == b.allocated;
    }

    inline friend bool operator!=(const VecSimAllocator &a, const VecSimAllocator &b) {
        return a.allocated != b.allocated;
    }

    static void setMemoryFunctions(VecSimMemoryFunctions memFunctions);

private:
    // Retrive the original requested allocation size. Required for remalloc.
    inline size_t getPointerAllocationSize(void *p) { return *(((size_t *)p) - 1); }
};

/**
 * @brief Global function to call for allocating memory buffer (malloc style).
 *
 * @param n - Amount of bytes to allocate.
 * @return void* - Allocated buffer.
 */
inline void *vecsim_malloc(size_t n) { return VecSimAllocator::memFunctions.allocFunction(n); }

/**
 * @brief Global function to call for allocating memory buffer initiliazed to zero (calloc style).
 *
 * @param nelem Number of elements.
 * @param elemsz Element size.
 * @return void* - Allocated buffer.
 */
inline void *vecsim_calloc(size_t nelem, size_t elemsz) {
    return VecSimAllocator::memFunctions.callocFunction(nelem, elemsz);
}

/**
 * @brief Global function to reallocate a buffer (realloc style).
 *
 * @param p Allocated buffer.
 * @param n Number of bytes required to the new buffer.
 * @return void* Allocated buffer with size >= n.
 */
inline void *vecsim_realloc(void *p, size_t n) {
    return VecSimAllocator::memFunctions.reallocFunction(p, n);
}

/**
 * @brief Global function to free an allocated buffer.
 *
 * @param p Allocated buffer.
 */
inline void vecsim_free(void *p) { VecSimAllocator::memFunctions.freeFunction(p); }

template <typename T>
struct VecsimSTLAllocator {
    using value_type = T;

private:
    VecsimSTLAllocator() {}

public:
    std::shared_ptr<VecSimAllocator> vecsim_allocator;
    VecsimSTLAllocator(std::shared_ptr<VecSimAllocator> vecsim_allocator)
        : vecsim_allocator(vecsim_allocator) {}

    // Copy constructor and assignment operator. Any VecsimSTLAllocator can be used for any type.
    template <typename U>
    VecsimSTLAllocator(const VecsimSTLAllocator<U> &other)
        : vecsim_allocator(other.vecsim_allocator) {}

    template <typename U>
    VecsimSTLAllocator &operator=(const VecsimSTLAllocator<U> &other) {
        this->vecsim_allocator = other.vecsim_allocator;
        return *this;
    }

    T *allocate(size_t size) { return (T *)this->vecsim_allocator->allocate(size * sizeof(T)); }

    void deallocate(T *ptr, size_t size) {
        this->vecsim_allocator->deallocate(ptr, size * sizeof(T));
    }
};

template <class T, class U>
bool operator==(const VecsimSTLAllocator<T> &a, const VecsimSTLAllocator<U> &b) {
    return a.vecsim_allocator == b.vecsim_allocator;
}
template <class T, class U>
bool operator!=(const VecsimSTLAllocator<T> &a, const VecsimSTLAllocator<U> &b) {
    return a.vecsim_allocator != b.vecsim_allocator;
}
