#include "vecsim_malloc.h"
#include <new>
void *operator new(size_t size) {
    void *p = vecsim_malloc(size);
    return p;
}

void *operator new[](size_t size) {
    void *p = vecsim_malloc(size);
    return p;
}

void operator delete(void *p) { vecsim_free(p); }

void operator delete[](void *p) { vecsim_free(p); }
