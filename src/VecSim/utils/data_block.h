#pragma once
#include <stddef.h>
#include <limits.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/vec_utils.h"

#define INVALID_ID UINT_MAX
typedef size_t labelType;
typedef size_t idType;

// Pre declaration

struct DataBlock;

struct DataBlockMember /*: public VecsimBaseObject*/ {
public:
    // DataBlockMember(std::shared_ptr<VecSimAllocator> allocator);
    union {
        void *vector;
        labelType label;
    };
    DataBlock *block;
    size_t index;
};

struct DataBlock : public VecsimBaseObject {

public:
    DataBlock(size_t blockSize, size_t elementSize, std::shared_ptr<VecSimAllocator> allocator,
              bool ownMembers = true);

    void addData(DataBlockMember *member, const void *data, idType id);

    inline void *getData(size_t index) {
        return (int8_t *)this->Data + (index * this->elementSize);
    }

    inline void *removeAndFetchData() {
        return (int8_t *)this->Data + ((--this->length) * this->elementSize);
    }

    inline size_t getLength() { return length; }

    inline idType getMember(size_t index) { return members ? this->members[index] : INVALID_ID; }

    inline void setMember(size_t index, DataBlockMember *member, idType id) {
        if (members) {
            this->members[index] = id;
            member->index = index;
            member->block = this;
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
    idType *members;
    // Data hosted in the data block.
    void *Data;
};
