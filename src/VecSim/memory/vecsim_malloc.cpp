#include "vecsim_malloc.h"
#include <stdlib.h>
#include <memory>

void *VecSimAllocator::allocate(size_t size) {
    *this->allocated.get() += size;
    return vecsim_malloc(size);
}

void VecSimAllocator::deallocate(void *p, size_t size) {
    *this->allocated.get() -= size;
    vecsim_free(p);
}

void *VecSimAllocator::operator new(size_t size) { return vecsim_malloc(size); }

void *VecSimAllocator::operator new[](size_t size) { return vecsim_malloc(size); }
void VecSimAllocator::operator delete(void *p, size_t size) { vecsim_free(p); }
void VecSimAllocator::operator delete[](void *p, size_t size) { vecsim_free(p); }

int64_t VecSimAllocator::getAllocationSize() { return *this->allocated.get(); }
