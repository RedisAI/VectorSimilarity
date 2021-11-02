#pragma once
#include <stddef.h>
#include <vector>
#include <queue>

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

using CandidatesHeap = std::priority_queue<std::pair<float, labelType>, std::vector<std::pair<float , labelType>>,
        CompareByFirst>;

struct VectorBlockMember {
public:
    size_t index;
    VectorBlock *block;
    labelType label;
};

struct VectorBlock {

public:
    VectorBlock(size_t blockSize, size_t vectorSize);

    void addVector(VectorBlockMember *vectorBlockMember, const void *vectorData);

    inline float *getVector(size_t index) { return this->vectors + (index * this->dim); }

    inline float *removeAndFetchVector() { return this->vectors + (this->size-- * this->dim); }

    inline size_t getSize() { return size; }

    inline VectorBlockMember *getMember(size_t index) { return this->members[index]; }

    inline void setMember(size_t index, VectorBlockMember *member) {
        this->members[index] = member;
    }

    std::vector<float> ComputeScores(DISTFUNC<float> DistFunc, const void *queryBlob);

    void heapBasedSearch(const std::vector<float> &scores, float lowerBound, float &upperBound,
                                      size_t nRes, CandidatesHeap &candidates);
    virtual ~VectorBlock();

private:
    size_t dim;
    size_t size;
    VectorBlockMember **members;
    float *vectors;
};
