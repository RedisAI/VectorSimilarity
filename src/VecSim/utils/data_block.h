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
    labelType label;
};

struct DataBlock : public VecsimBaseObject {

public:
    DataBlock(size_t blockSize, size_t unitSize, std::shared_ptr<VecSimAllocator> allocator);

    void addData(DataBlockMember *dataBlockMember, const void *data);

    inline void *getData(size_t index) { return (int8_t *)this->Data + (index * this->unitSize); }

    inline void *removeAndFetchData() {
        return (int8_t *)this->Data + ((--this->length) * this->unitSize);
    }

    inline size_t getLength() { return length; }

    inline DataBlockMember *getMember(size_t index) { return this->members[index]; }

    inline void setMember(size_t index, DataBlockMember *member) {
        this->members[index] = member;
        member->index = index;
        member->block = this;
    }

    virtual ~DataBlock();

private:
    // Data unit size.
    size_t unitSize;
    // Current data block length.
    size_t length;
    // Data block size (capacity).
    size_t blockSize;
    // Current members of the data block.
    DataBlockMember **members;
    // Data hosted in the data block.
    void *Data;
};
