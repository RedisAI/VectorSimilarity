#include "vecsim_base.h"

void *VecsimBaseObject::operator new(size_t size, VecSimAllocator &allocator) {
    return allocator.allocate(size);
}

void *VecsimBaseObject::operator new(size_t size, VecSimAllocator *allocator) {
    return allocator->allocate(size);
}

void *VecsimBaseObject::operator new(size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    return allocator->allocate(size);
}

void *VecsimBaseObject::operator new[](size_t size, VecSimAllocator &allocator) {
    return allocator.allocate(size);
}

void *VecsimBaseObject::operator new[](size_t size, VecSimAllocator *allocator) {
    return allocator->allocate(size);
    ;
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

void operator delete(void *p, size_t size, VecSimAllocator *allocator) {
    allocator->deallocate(p, size);
}

void operator delete[](void *p, size_t size, VecSimAllocator *allocator) {
    allocator->deallocate(p, size);
}

void operator delete(void *p, size_t size, VecSimAllocator &allocator) {
    allocator.deallocate(p, size);
}

void operator delete[](void *p, size_t size, VecSimAllocator &allocator) {
    allocator.deallocate(p, size);
}

void operator delete(void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->deallocate(p, size);
}

void operator delete[](void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator) {
    allocator->deallocate(p, size);
}

std::shared_ptr<VecSimAllocator> VecsimBaseObject::getAllocator() { return this->allocator; }
