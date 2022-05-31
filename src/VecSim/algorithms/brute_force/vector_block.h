#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/vec_utils.h"

typedef size_t labelType;
typedef size_t idType;

// Pre declaration

struct VectorBlock;

struct VectorBlockMember : public VecsimBaseObject {
public:
    VectorBlockMember(std::shared_ptr<VecSimAllocator> allocator);
    size_t index;
    VectorBlock *block;
    labelType label;
};

struct VectorBlock : public VecsimBaseObject {

public:
    VectorBlock(size_t blockSize, size_t vectorSize, std::shared_ptr<VecSimAllocator> allocator);

    void addVector(VectorBlockMember *vectorBlockMember, const void *vectorData);

    inline float *getVector(size_t index) { return this->vectors + (index * this->dim); }

    inline float *removeAndFetchVector() { return this->vectors + ((--this->length) * this->dim); }

    inline size_t getLength() { return length; }

    inline VectorBlockMember *getMember(size_t index) { return this->members[index]; }

    inline void setMember(size_t index, VectorBlockMember *member) {
        this->members[index] = member;
        member->index = index;
        member->block = this;
    }

    virtual ~VectorBlock();

private:
    // Vector dimensions.
    size_t dim;
    // Current vector block length.
    size_t length;
    // Vector block size (capacity).
    size_t blockSize;
    // Current members of the vector block.
    VectorBlockMember **members;
    // Vectors hosted in the vector block.
    float *vectors;
};
