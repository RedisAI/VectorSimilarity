/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "vector_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>

VectorBlock::VectorBlock(size_t blockSize, size_t vectorBytesCount,
                         std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator), vector_bytes_count(vectorBytesCount), length(0),
      blockSize(blockSize) {
    this->vectors = (char *)this->allocator->allocate(vectorBytesCount * blockSize);
}

VectorBlock::~VectorBlock() {
    this->allocator->deallocate(vectors, vector_bytes_count * blockSize);
}

void VectorBlock::addVector(const void *vectorData) {

    // Copy vector data and update block size.
    memcpy(this->vectors + (this->length * vector_bytes_count), vectorData, vector_bytes_count);
    this->length++;
}

void VectorBlock::updateVector(size_t index, const void *vector_data) {
    char *destinaion = getVector(index);
    memcpy(destinaion, vector_data, vector_bytes_count);
}
