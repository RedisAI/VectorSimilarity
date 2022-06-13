#include "data_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>

// DataBlockMember::DataBlockMember(std::shared_ptr<VecSimAllocator> allocator)
//     : VecsimBaseObject(allocator) {}

DataBlock::DataBlock(size_t blockSize, size_t elementSize,
                     std::shared_ptr<VecSimAllocator> allocator, bool ownMembers)
    : VecsimBaseObject(allocator), elementSize(elementSize), length(0), blockSize(blockSize) {
    if (ownMembers) {
        this->members = (idType *)this->allocator->allocate(sizeof(idType) * blockSize);
    } else {
        this->members = NULL;
    }
    this->Data = this->allocator->allocate(blockSize * elementSize);
}

DataBlock::~DataBlock() {
    if (members) {
        this->allocator->deallocate(members, sizeof(idType) * blockSize);
    }
    this->allocator->deallocate(Data, blockSize * elementSize);
}

void DataBlock::addData(DataBlockMember *member, const void *data, idType id) {
    // Mutual point both structs on each other.
    setMember(this->length, member, id);

    // Copy data and update block size.
    memcpy((int8_t *)this->Data + (this->length * this->elementSize), data, this->elementSize);
    this->length++;
}
