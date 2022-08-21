#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/utils/vec_utils.h"

struct VectorBlock : public VecsimBaseObject {

public:
    VectorBlock(size_t blockSize, size_t vectorSize, std::shared_ptr<VecSimAllocator> allocator);

    void addVector(const void *vectorData);

    void updateVector(size_t index, const void *vector_data);

    inline float *getVector(size_t index) { return this->vectors + (index * this->dim); }

    inline float *removeAndFetchLastVector() {
        return this->vectors + ((--this->length) * this->dim);
    }

    inline size_t getLength() { return length; }

    virtual ~VectorBlock();

private:
    // Vector dimensions.
    size_t dim;
    // Current vector block length.
    size_t length;
    // Vector block size (capacity).
    size_t blockSize;
    // Vectors hosted in the vector block.
    float *vectors;
};
