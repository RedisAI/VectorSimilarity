#pragma once
#include <stddef.h>
#include "vec_sim_common.h"

struct VecSimIndexTombstone {
protected:
    size_t num_marked_deleted;

public:
    VecSimIndexTombstone() : num_marked_deleted(0) {}
    ~VecSimIndexTombstone() = default;

    inline size_t getNumMarkedDeleted() const { return num_marked_deleted; }

    virtual inline int markDelete(labelType label) = 0;
    virtual inline int unmarkDelete(labelType label) = 0;
};
