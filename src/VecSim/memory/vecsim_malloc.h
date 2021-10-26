#pragma once

#include "redismodule.h"
#include <stddef.h>
#include <memory>

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
template <typename T>
struct VecsimAllocator {
    using value_type = T;

    std::shared_ptr<uint64_t> allocated;

    VecsimAllocator() : allocated(new uint64_t) { *allocated = sizeof(allocated); }

    template <typename U>
    VecsimAllocator(const VecsimAllocator<U> &other) : allocated(other.allocated) {}

    template <typename U>
    VecsimAllocator &operator=(const VecsimAllocator<U> &other) {
        std::allocator<T>::operator=(other);
        this->allocated = other.allocated;
        return *this;
    }

    T *allocate(size_t size) {

        *allocated.get() += size * sizeof(T);
        return (T *)vecsim_malloc(size * sizeof(T));
    }

    void deallocate(T *ptr, size_t size) {
        *allocated.get() -= size * sizeof(T);
        vecsim_free(ptr);
    }

    T *allocate_blocks(size_t size) {

        *allocated.get() += size;
        return (T *)vecsim_malloc(size);
    }

    void deallocate_blocks(T *ptr, size_t size) {
        *allocated.get() -= size;
        vecsim_free(ptr);
    }
};

template <class T, class U>
bool operator==(const VecsimAllocator<T> &, const VecsimAllocator<U> &) {
    return true;
}
template <class T, class U>
bool operator!=(const VecsimAllocator<T> &, const VecsimAllocator<U> &) {
    return false;
}

struct VecsimBaseObject {

private:
    VecsimAllocator<VecsimBaseObject> allocator;

public:
    VecsimBaseObject() {}
    VecsimBaseObject(VecsimAllocator<VecsimBaseObject> allocator) : allocator(allocator) {}

    void *operator new(size_t size, VecsimAllocator<VecsimBaseObject> &allocator);
    void *operator new[](size_t size, VecsimAllocator<VecsimBaseObject> &allocator);
    void operator delete(void *p, VecsimAllocator<VecsimBaseObject> &allocator);
    void operator delete[](void *p, VecsimAllocator<VecsimBaseObject> &allocator);
    virtual size_t size() = 0;
    void setAllocator(VecsimAllocator<VecsimBaseObject> &allocator);
    VecsimAllocator<VecsimBaseObject> &getAllocator();
};
