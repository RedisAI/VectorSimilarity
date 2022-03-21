#include "data_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>

DataBlockMember::DataBlockMember(std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator) {}

DataBlock::DataBlock(size_t blockSize, size_t unitSize, std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator), unitSize(unitSize), length(0), blockSize(blockSize) {
    this->members =
        (DataBlockMember **)this->allocator->allocate(sizeof(DataBlockMember *) * blockSize);
    this->Data = this->allocator->allocate(blockSize * unitSize);
}

DataBlock::~DataBlock() {
    for (size_t i = 0; i < this->length; i++) {
        delete members[i];
    }
    this->allocator->deallocate(members, sizeof(DataBlockMember *) * blockSize);
    this->allocator->deallocate(Data, blockSize * unitSize);
}

void DataBlock::addData(DataBlockMember *dataBlockMember, const void *data) {
    // Mutual point both structs on each other.
    this->members[this->length] = dataBlockMember;
    dataBlockMember->block = this;
    dataBlockMember->index = this->length;

    // Copy data and update block size.
    memcpy((int8_t *)this->Data + (this->length * this->unitSize), data, this->unitSize);
    this->length++;
}
