#include "vector_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>


VectorBlock::VectorBlock(size_t blockSize, size_t vectorSize,
                         std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator), dim(vectorSize), length(0), blockSize(blockSize) {
    this->vectors = (float *)this->allocator->allocate(sizeof(float) * blockSize * vectorSize);
}

VectorBlock::~VectorBlock() {

    this->allocator->deallocate(vectors, sizeof(float) * blockSize * dim);
}

void VectorBlock::addVector(VectorBlockMember *vectorBlockMember, const void *vectorData) {
    // Mutual point both structs on each other.
    this->members[this->length] = vectorBlockMember;
    vectorBlockMember->block = this;
    vectorBlockMember->index = this->length;

    // Copy vector data and update block size.
    memcpy(this->vectors + (this->length * this->dim), vectorData, this->dim * sizeof(float));
    this->length++;
}

void VectorBlock::updateVector(size_t index, const void *vector_data)
{
    float *destinaion = getVector(index);
    memcpy(destinaion, vector_data, this->dim);

}
