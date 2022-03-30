#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/vec_utils.h"

typedef size_t labelType;
typedef size_t idType;

// Pre declaration

struct DataBlock;

struct DataBlockMember : public VecsimBaseObject {
public:
    DataBlockMember(std::shared_ptr<VecSimAllocator> allocator);
    size_t index;
    DataBlock *block;
    void *vector;
    labelType label;
};

struct DataBlock : public VecsimBaseObject {

public:
    DataBlock(size_t blockSize, size_t elementSize, std::shared_ptr<VecSimAllocator> allocator,
              size_t index = -1, bool ownMembers = true);

    void addData(DataBlockMember *dataBlockMember, const void *data);

    inline void *getData(size_t index) {
        return (int8_t *)this->Data + (index * this->elementSize);
    }

    inline void *removeAndFetchData() {
        return (int8_t *)this->Data + ((--this->length) * this->elementSize);
    }

    inline size_t getLength() { return length; }

    inline size_t getIndex() { return index; }

    inline DataBlockMember *getMember(size_t index) {
        return members ? this->members[index] : NULL;
    }

    inline void setMember(size_t index, DataBlockMember *member) {
        if (members) {
            this->members[index] = member;
            member->index = index;
            member->block = this;
        } else {
            member->vector = getData(index);
        }
    }

    virtual ~DataBlock();

private:
    // Data element size (in bytes).
    size_t elementSize;
    // Current data block length.
    size_t length;
    // Data block size (capacity).
    size_t blockSize;
    // Current members of the data block.
    DataBlockMember **members;
    // Index block in vector of blocks
    size_t index;
    // Data hosted in the data block.
    void *Data;
};
