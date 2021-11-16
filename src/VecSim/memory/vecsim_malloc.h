#pragma once
#include <stddef.h>
#include <memory>

#ifdef REDIS_MODULE_TARGET /* Set this when compiling your code as a module */
#include "redismodule.h"
#include <cstring>
static inline void *vecsim_malloc(size_t n) { return RedisModule_Alloc(n); }
static inline void *vecsim_calloc(size_t nelem, size_t elemsz) {
    return RedisModule_Calloc(nelem, elemsz);
}
static inline void *vecsim_realloc(void *p, size_t n) { return RedisModule_Realloc(p, n); }
static inline void vecsim_free(void *p) { RedisModule_Free(p); }
static inline char *vecsim_strdup(const char *s) { return RedisModule_Strdup(s); }

static inline char *vecsim_strndup(const char *s, size_t n) {
    char *ret = (char *)vecsim_malloc(n + 1);

    if (ret) {
        ret[n] = '\0';
        memcpy(ret, s, n);
    }
    return ret;
}
#else

#define vecsim_malloc  malloc
#define vecsim_free    free
#define vecsim_calloc  calloc
#define vecsim_realloc realloc
#define vecsim_strdup  strdup
#define vecsim_strndup strndup

#endif

struct VecSimAllocator {
private:
    std::shared_ptr<uint64_t> allocated;

    // Static member that indicates each allocation additional size.
    static size_t allocation_header_size;

    VecSimAllocator() : allocated(std::make_shared<uint64_t>(sizeof(VecSimAllocator))) {}

public:
    static std::shared_ptr<VecSimAllocator> newVecsimAllocator();
    void *allocate(size_t size);
    void deallocate(void *p, size_t size);
    void *reallocate(void *p, size_t size);
    void free_allocation(void *p);

    void *operator new(size_t size);
    void *operator new[](size_t size);
    void operator delete(void *p, size_t size);
    void operator delete[](void *p, size_t size);

    int64_t getAllocationSize();
    inline friend bool operator==(const VecSimAllocator &a, const VecSimAllocator &b) {
        return a.allocated == b.allocated;
    }

    inline friend bool operator!=(const VecSimAllocator &a, const VecSimAllocator &b) {
        return a.allocated != b.allocated;
    }

private:
    // Retrive the original requested allocation size. Required for remalloc.
    inline size_t getPointerAllocationSize(void *p) { return *(((size_t *)p) - 1); }
};

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
