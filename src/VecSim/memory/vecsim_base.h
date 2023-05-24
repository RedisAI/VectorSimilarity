/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "vecsim_malloc.h"
#include <memory>

struct VecsimBaseObject {

protected:
    std::shared_ptr<VecSimAllocator> allocator;

public:
    VecsimBaseObject(std::shared_ptr<VecSimAllocator> allocator) : allocator(allocator) {}

    static void *operator new(size_t size, std::shared_ptr<VecSimAllocator> allocator);
    static void *operator new[](size_t size, std::shared_ptr<VecSimAllocator> allocator);
    static void operator delete(void *p, size_t size);
    static void operator delete[](void *p, size_t size);

    // Placement delete. To be used in try/catch clause when called with the respected constructor
    static void operator delete(void *p, std::shared_ptr<VecSimAllocator> allocator);
    static void operator delete[](void *p, std::shared_ptr<VecSimAllocator> allocator);
    static void operator delete(void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator);
    static void operator delete[](void *p, size_t size, std::shared_ptr<VecSimAllocator> allocator);

    std::shared_ptr<VecSimAllocator> getAllocator() const;
    virtual inline uint64_t getAllocationSize() const {
        return this->allocator->getAllocationSize();
    }

    virtual ~VecsimBaseObject() {}
};
