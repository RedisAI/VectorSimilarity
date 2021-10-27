#pragma once
#include "vecsim_malloc.h"
#include <memory>

struct VecsimBaseObject {

private:
    std::shared_ptr<VecSimAllocator> allocator;

public:
    VecsimBaseObject() {}
    VecsimBaseObject(VecSimAllocator *allocator) : allocator(allocator) {}
    VecsimBaseObject(std::shared_ptr<VecSimAllocator> allocator) : allocator(allocator) {}

    void *operator new(size_t size, VecSimAllocator &allocator);
    void *operator new(size_t size, VecSimAllocator *allocator);
    void *operator new(size_t size, std::shared_ptr<VecSimAllocator> allocator);
    void *operator new[](size_t size, VecSimAllocator &allocator);
    void *operator new[](size_t size, VecSimAllocator *allocator);
    void *operator new[](size_t size, std::shared_ptr<VecSimAllocator> allocator);
    void operator delete(void *p, size_t size);
    void operator delete[](void *p, size_t size);
    std::shared_ptr<VecSimAllocator> getAllocator();
};
