/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "data_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>

DataBlock::DataBlock(size_t blockSize, size_t elementBytesCount,
                     std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator), element_bytes_count(elementBytesCount), length(0),
      data((char *)this->allocator->allocate(blockSize * elementBytesCount)) {}

DataBlock::~DataBlock() { this->allocator->free_allocation(data); }

void DataBlock::addElement(const void *element) {

    // Copy element data and update block size.
    memcpy(this->data + (this->length * element_bytes_count), element, element_bytes_count);
    this->length++;
}

void DataBlock::updateElement(size_t index, const void *new_element) {
    char *destinaion = getElement(index);
    memcpy(destinaion, new_element, element_bytes_count);
}
