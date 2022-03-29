#include "data_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>

DataBlockMember::DataBlockMember(std::shared_ptr<VecSimAllocator> allocator)
    : VecsimBaseObject(allocator) {}

DataBlock::DataBlock(size_t blockSize, size_t elementSize,
                     std::shared_ptr<VecSimAllocator> allocator, size_t index, bool ownMembers)
    : VecsimBaseObject(allocator), elementSize(elementSize), length(0), blockSize(blockSize),
      index(index), membersOwner(ownMembers) {
    this->members =
        (DataBlockMember **)this->allocator->allocate(sizeof(DataBlockMember *) * blockSize);
    this->Data = this->allocator->allocate(blockSize * elementSize);
}

DataBlock::~DataBlock() {
    if (membersOwner) {
        for (size_t i = 0; i < this->length; i++) {
            delete members[i];
        }
    }
    this->allocator->deallocate(members, sizeof(DataBlockMember *) * blockSize);
    this->allocator->deallocate(Data, blockSize * elementSize);
}

void DataBlock::addData(DataBlockMember *dataBlockMember, const void *data) {
    // Mutual point both structs on each other.
    setMember(this->length, dataBlockMember);

    // Copy data and update block size.
    memcpy((int8_t *)this->Data + (this->length * this->elementSize), data, this->elementSize);
    this->length++;
}
