#pragma once
#include <stddef.h>
#include <memory>
#include "redismodule.h"

#ifdef REDIS_MODULE_TARGET /* Set this when compiling your code as a module */
static inline void *vecsim_malloc(size_t n) { return RedisModule_Alloc(n); }
static inline void *vecsim_calloc(size_t nelem, size_t elemsz) {
    return RedisModule_Calloc(nelem, elemsz);
}
static inline void *vecsim_realloc(void *p, size_t n) { return RedisModule_Realloc(p, n); }
static inline void vecsim_free(void *p) { RedisModule_Free(p); }
static inline char *vecsim_strdup(const char *s) { return RedisModule_Strdup(s); }

static inline char *vecsim_strndup(const char *s, size_t n) {
    char *ret = (char *)rm_malloc(n + 1);

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
    size_t allocation_header_size;

public:
    VecSimAllocator()
        : allocated(std::make_shared<uint64_t>(sizeof(VecSimAllocator))),
          allocation_header_size(sizeof(size_t)) {}

    void *allocate(size_t size);
    void deallocate(void *p, size_t size);
    void *reallocate(void *p, size_t size);
    void free_allocation(void *p);

    void *operator new(size_t size);
    void *operator new[](size_t size);
    void operator delete(void *p, size_t size);
    void operator delete[](void *p, size_t size);

    int64_t getAllocationSize();

private:
    inline size_t getPointerAllocationSize(void *p) { return *(((size_t *)p) - 1); }
};

template <typename T>
struct VecsimSTLAllocator {
    using value_type = T;

    std::shared_ptr<VecSimAllocator> vecsim_allocator;
    VecsimSTLAllocator() {}
    VecsimSTLAllocator(std::shared_ptr<VecSimAllocator> vecsim_allocator)
        : vecsim_allocator(vecsim_allocator) {}
    VecsimSTLAllocator(VecSimAllocator *vecsim_allocator) : vecsim_allocator(vecsim_allocator) {}
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
bool operator==(const VecsimSTLAllocator<T> &, const VecsimSTLAllocator<U> &) {
    return true;
}
template <class T, class U>
bool operator!=(const VecsimSTLAllocator<T> &, const VecsimSTLAllocator<U> &) {
    return false;
}
