#pragma once

#include "space_interface.h"

namespace hnswlib {
class L2Space : public SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

  public:
    explicit L2Space(size_t dim);
    ~L2Space() override;

    size_t get_data_size() const override;
    DISTFUNC<float> get_dist_func() const override;
    void *get_data_dim() override;
};
} // namespace hnswlib
