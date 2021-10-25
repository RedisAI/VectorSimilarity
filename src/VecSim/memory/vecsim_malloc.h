#pragma once

#include "redismodule.h"
#include <stddef.h>

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
struct vecsim_allocator {
    using value_type = T;

    template <typename U>
    vecsim_allocator(const vecsim_allocator<U> &other) {}

    T *allocate(size_t size) { return vecsim_malloc(size * sizeof(T)); }

    void deallocate(T *ptr, size_t size) { vecsim_free(ptr); }
};

struct VecsimBaseObject {
	public:
	void * operator new(size_t size);
    void operator delete(void * p);
};
