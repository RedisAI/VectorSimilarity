#pragma once

#include "space_interface.h"

#include <cstdlib>
#include <stddef.h>

class InnerProductSpace : public SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    explicit InnerProductSpace(size_t dim, std::shared_ptr<VecSimAllocator> allocator);
    ~InnerProductSpace();

    size_t get_data_size() const;
    DISTFUNC<float> get_dist_func() const;
    void *get_data_dim();
};
