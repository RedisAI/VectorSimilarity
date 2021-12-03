
#pragma once
#include "VecSim/memory/vecsim_base.h"

#include <stddef.h>

template <typename TYPE>
using DISTFUNC = TYPE (*)(const void *, const void *, const void *);

template <typename TYPE>
class SpaceInterface : public VecsimBaseObject {
public:
    SpaceInterface(std::shared_ptr<VecSimAllocator> allocator) : VecsimBaseObject(allocator) {}

    virtual size_t get_data_size() const = 0;

    virtual DISTFUNC<TYPE> get_dist_func() const = 0;

    virtual void *get_data_dim() = 0;

    virtual ~SpaceInterface() = default;
};
