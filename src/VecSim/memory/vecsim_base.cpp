/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "vecsim_base.h"

void *VecsimBaseObject::operator new(size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    return allocator->allocate(size);
}

void *VecsimBaseObject::operator new[](size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    return allocator->allocate(size);
}

void VecsimBaseObject::operator delete(void *p, size_t size) {
    VecsimBaseObject *obj = reinterpret_cast<VecsimBaseObject *>(p);
    obj->allocator->deallocate(obj, size);
}

void VecsimBaseObject::operator delete[](void *p, size_t size) {
    VecsimBaseObject *obj = reinterpret_cast<VecsimBaseObject *>(p);
    obj->allocator->deallocate(obj, size);
}

void VecsimBaseObject::operator delete(void *p, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->free_allocation(p);
}
void VecsimBaseObject::operator delete[](void *p, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->free_allocation(p);
}

void operator delete(void *p, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->free_allocation(p);
}
void operator delete[](void *p, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->free_allocation(p);
}

// TODO: Probably unused functions. See Codcove output in order to remove

void operator delete(void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->deallocate(p, size);
}

void operator delete[](void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->deallocate(p, size);
}

std::shared_ptr<VecSimAllocator> VecsimBaseObject::getAllocator() const { return this->allocator; }
