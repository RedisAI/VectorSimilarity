#include "vector_block.h"
#include <cstring>
#include <vector>

VectorBlock::VectorBlock(size_t blockSize, size_t vectorSize) : dim(vectorSize), size(0) {
    this->members = new VectorBlockMember *[blockSize];
    this->vectors = new float[blockSize * vectorSize];
}

VectorBlock::~VectorBlock() {
    for (size_t i = 0; i < this->size; i++) {
        delete members[i];
    }
    delete[] members;
    delete[] vectors;
}

void VectorBlock::addVector(VectorBlockMember *vectorBlockMember, const void *vectorData) {
    // Mutual point both structs on each other.
    this->members[this->size] = vectorBlockMember;
    vectorBlockMember->block = this;
    vectorBlockMember->index = this->size;

    // Copy vector data and update block size.
    memcpy(this->vectors + (this->size * this->dim), vectorData, this->dim * sizeof(float));
    this->size++;
}

std::vector<std::pair<float, labelType>> VectorBlock::computeBlockScores(DISTFUNC<float> DistFunc,
                                                                         const void *queryBlob) {
    std::vector<std::pair<float, labelType>> scores(size);
    for (size_t i = 0; i < size; i++) {
        scores[i] = {DistFunc(this->getVector(i), queryBlob, &dim), getMember(i)->label};
    }
    return scores;
}
