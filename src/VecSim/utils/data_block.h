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
              std::shared_ptr<VecSimAllocator> allocator);
    DataBlock(DataBlock &&other) noexcept;
    ~DataBlock() noexcept;

    DataBlock(const DataBlock &other) = delete;

    DataBlock &operator=(DataBlock &&other) noexcept {
        element_bytes_count = other.element_bytes_count;
        length = other.length;
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
    // Element size in bytes (dim * sizeof(data_type))
    size_t element_bytes_count;
    // Current block length.
    size_t length;
    // Elements hosted in the block.
    char *data;
};
