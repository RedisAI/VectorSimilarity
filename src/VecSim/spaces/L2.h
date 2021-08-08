#pragma once

#include "space_interface.h"

class L2Space : public SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

  public:
    L2Space(size_t dim);
    ~L2Space();

    size_t get_data_size() const;
    DISTFUNC<float> get_dist_func() const;
    void *get_data_dim();
};
