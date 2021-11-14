#include "vecsim_malloc.h"
#include <stdlib.h>
#include <memory>
#include <string.h>

size_t VecSimAllocator::allocation_header_size = sizeof(size_t);

void *VecSimAllocator::allocate(size_t size) {
    *this->allocated.get() += size + allocation_header_size;
    size_t *ptr = (size_t *)vecsim_malloc(size + allocation_header_size);
    *ptr = size;

    return ptr + 1;
}

void VecSimAllocator::deallocate(void *p, size_t size) { free_allocation(p); }

void *VecSimAllocator::reallocate(void *p, size_t size) {
    size_t *ptr = ((size_t *)p) - 1;
    size_t oldSize = getPointerAllocationSize(p);
    if (oldSize >= size) {
        return p;
    }
    void *new_ptr = this->allocate(size);
    memcpy(new_ptr, p, oldSize);
    free_allocation(p);
    return new_ptr;
}

void VecSimAllocator::free_allocation(void *p) {
    if (!p)
        return;
    size_t *ptr = ((size_t *)p) - 1;
    *this->allocated.get() -= ptr[0];
    vecsim_free(ptr);
}

void *VecSimAllocator::operator new(size_t size) { return vecsim_malloc(size); }

void *VecSimAllocator::operator new[](size_t size) { return vecsim_malloc(size); }
void VecSimAllocator::operator delete(void *p, size_t size) { vecsim_free(p); }
void VecSimAllocator::operator delete[](void *p, size_t size) { vecsim_free(p); }

int64_t VecSimAllocator::getAllocationSize() { return *this->allocated.get(); }
