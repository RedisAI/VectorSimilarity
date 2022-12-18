/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/utils/vec_utils.h"

struct VectorBlock : public VecsimBaseObject {

public:
    VectorBlock(size_t blockSize, size_t vectorBytesCount,
                std::shared_ptr<VecSimAllocator> allocator);

    void addVector(const void *vectorData);

    void updateVector(size_t index, const void *vector_data);

    inline char *getVector(size_t index) { return this->vectors + (index * vector_bytes_count); }

    inline char *removeAndFetchLastVector() {
        return this->vectors + ((--this->length) * vector_bytes_count);
    }

    inline size_t getLength() { return length; }

    virtual ~VectorBlock();

private:
    // Vector size in bytes (dim * sizeof(data_type))
    size_t vector_bytes_count;
    // Current vector block length.
    size_t length;
    // Vector block size (capacity).
    size_t blockSize;
    // Vectors hosted in the vector block.
    char *vectors;
};
