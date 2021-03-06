#pragma once

#include "VecSim/spaces/space_interface.h"

class L2Space : public SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    explicit L2Space(size_t dim, std::shared_ptr<VecSimAllocator> allocator);
    ~L2Space() override;

    size_t get_data_size() const override;
    DISTFUNC<float> get_dist_func() const override;
    void *get_data_dim() override;
};
