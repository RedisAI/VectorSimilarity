/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vecsim_stl.h"

#include "VecSim/utils/vec_utils.h"

struct DataBlock : public VecsimBaseObject {

public:
    DataBlock(size_t blockSize, size_t elementBytesCount,
              std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment = 0);
    // Move constructor
    // We need to implement this because we want to have a vector of DataBlocks, and we want it to
    // use the move constructor upon resizing (instead of the copy constructor). We also need to
    // mark it as noexcept so the vector will use it.
    DataBlock(DataBlock &&other) noexcept;
    ~DataBlock() noexcept;
    // Delete copy constructor so we won't have a vector of DataBlocks that uses the copy
    // constructor
    DataBlock(const DataBlock &other) = delete;

    DataBlock &operator=(DataBlock &&other) noexcept {
        allocator = other.allocator;
        element_bytes_count = other.element_bytes_count;
        length = other.length;
        // take ownership of the data
        data = other.data;
        other.data = nullptr;
        return *this;
    };

    void addElement(const void *element);

    void updateElement(size_t index, const void *new_element);

    inline const char *getElement(size_t index) const {
        return this->data + (index * element_bytes_count);
    }

    inline char *removeAndFetchLastElement() {
        return this->data + ((--this->length) * element_bytes_count);
    }

    inline size_t getLength() const { return length; }

private:
    // Element size in bytes
    size_t element_bytes_count;
    // Current block length.
    size_t length;
    // Elements hosted in the block.
    char *data;
};
