#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/spaces/space_interface.h"

typedef size_t labelType;
typedef size_t idType;

// Pre declaration

struct VectorBlock;

// TODO: unify this with HNSW
struct CompareByFirst {
    constexpr bool operator()(std::pair<float, labelType> const &a,
                              std::pair<float, labelType> const &b) const noexcept {
        return a.first < b.first;
    }
};

using CandidatesHeap =
    vecsim_stl::priority_queue<std::pair<float, labelType>,
                               vecsim_stl::vector<std::pair<float, labelType>>, CompareByFirst>;

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

    inline float *removeAndFetchVector() { return this->vectors + (this->length-- * this->dim); }

    inline size_t getLength() { return length; }

    inline VectorBlockMember *getMember(size_t index) { return this->members[index]; }

    inline void setMember(size_t index, VectorBlockMember *member) {
        this->members[index] = member;
    }

    // Compute the score for every vector in the block by using the given distance function.
    // Return a collection of (score, label) pairs for every vector in the block.
    vecsim_stl::vector<std::pair<float, labelType>> computeBlockScores(DISTFUNC<float> DistFunc,
                                                                       const void *queryBlob);

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
