#include "vecsim_malloc.h"
#include "memory"

// Global new/delete overload for non trackable objects

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

// Trackable objects allocation
void *VecsimBaseObject::operator new(size_t size, VecsimAllocator<VecsimBaseObject> &allocator) {
    void *p = allocator.allocate_blocks(size);
    return p;
}

void *VecsimBaseObject::operator new[](size_t size, VecsimAllocator<VecsimBaseObject> &allocator) {
    void *p = allocator.allocate_blocks(size);
    return p;
}

void VecsimBaseObject::operator delete(void *p, VecsimAllocator<VecsimBaseObject> &allocator) {
    VecsimBaseObject *obj = reinterpret_cast<VecsimBaseObject *>(p);
    allocator.deallocate_blocks(obj, obj->size());
}

void VecsimBaseObject::operator delete[](void *p, VecsimAllocator<VecsimBaseObject> &allocator) {
    VecsimBaseObject *obj = reinterpret_cast<VecsimBaseObject *>(p);
    allocator.deallocate_blocks(obj, obj->size());
}

void VecsimBaseObject::setAllocator(VecsimAllocator<VecsimBaseObject> &allocator) {
    this->allocator = allocator;
}

VecsimAllocator<VecsimBaseObject> &VecsimBaseObject::getAllocator() { return this->allocator; }
