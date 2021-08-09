#pragma once

#include <cstdlib>
#include "space_interface.h"

namespace hnswlib {
class InnerProductSpace : public SpaceInterface<float> {

    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

  public:
    explicit InnerProductSpace(size_t dim);
    ~InnerProductSpace();

    size_t get_data_size() const;
    DISTFUNC<float> get_dist_func() const;
    void *get_data_dim();
};
} // namespace hnswlib
