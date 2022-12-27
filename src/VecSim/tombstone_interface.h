#pragma once
#include <stddef.h>
#include "vec_sim_common.h"

/*
 * Defines a simple tombstone API for indexes.
 * Every index that has to implement "marking as deleted" mechanism should inherit this API and
 * implement the required functions. The implementation should also update the `num_marked_deleted`
 * property to hold the number of vectors marked as deleted.
 */
struct VecSimIndexTombstone {
protected:
    size_t num_marked_deleted;

public:
    VecSimIndexTombstone() : num_marked_deleted(0) {}
    ~VecSimIndexTombstone() = default;

    inline size_t getNumMarkedDeleted() const { return num_marked_deleted; }

    /**
     * @param label vector to mark as deleted
     * @return a vector of internal ids that has been marked as deleted (to be disposed later on).
     */
    virtual inline std::vector<idType> markDelete(labelType label) = 0;
};
