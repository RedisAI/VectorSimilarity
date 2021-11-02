#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"

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

    inline float *removeAndFetchVector() { return this->vectors + (this->size-- * this->dim); }

    inline size_t getSize() { return size; }

    inline VectorBlockMember *getMember(size_t index) { return this->members[index]; }

    inline void setMember(size_t index, VectorBlockMember *member) {
        this->members[index] = member;
    }

    virtual ~VectorBlock();

private:
    size_t dim;
    size_t size;
    size_t blockSize;
    VectorBlockMember **members;
    float *vectors;
};
