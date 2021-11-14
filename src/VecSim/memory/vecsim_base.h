#pragma once
#include "vecsim_malloc.h"
#include <memory>

struct VecsimBaseObject {

protected:
    std::shared_ptr<VecSimAllocator> allocator;

public:
    VecsimBaseObject(std::shared_ptr<VecSimAllocator> allocator) : allocator(allocator) {}

    void *operator new(size_t size, std::shared_ptr<VecSimAllocator> allocator);
    void *operator new[](size_t size, std::shared_ptr<VecSimAllocator> allocator);
    void operator delete(void *p, size_t size);
    void operator delete[](void *p, size_t size);

    // Placement delete. To be used in try/catch clause when called with the respected constructor
    void operator delete(void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator);
    void operator delete[](void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator);

    std::shared_ptr<VecSimAllocator> getAllocator();
    virtual ~VecsimBaseObject() {}
};
