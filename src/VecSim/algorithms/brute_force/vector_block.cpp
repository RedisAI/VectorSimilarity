#include "vector_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>

VectorBlockMember::VectorBlockMember(std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator) {}

VectorBlock::VectorBlock(size_t blockSize, size_t vectorSize,
                         std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator), dim(vectorSize), length(0), blockSize(blockSize) {
    this->members =
        (VectorBlockMember **)this->allocator->allocate(sizeof(VectorBlockMember *) * blockSize);
    this->vectors = (float *)this->allocator->allocate(sizeof(float) * blockSize * vectorSize);
}

VectorBlock::~VectorBlock() {
    for (size_t i = 0; i < this->length; i++) {
        delete members[i];
    }
    this->allocator->deallocate(members, sizeof(VectorBlockMember *) * blockSize);
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
