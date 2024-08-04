#pragma once

#include "data_block.h"
#include "raw_data_container_interface.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"

class DataBlocksContainer : public VecsimBaseObject, public RawDataContainer {

    size_t element_bytes_count;           // Element size in bytes
    size_t element_count;                 // Number of items in the container
    vecsim_stl::vector<DataBlock> blocks; // data blocks
    size_t block_size;                    // number of element in block
    unsigned char alignment;              // alignment for data allocationin each block

public:
    DataBlocksContainer(size_t blockSize, size_t elementBytesCount,
                        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment = 0);
    ~DataBlocksContainer();

    size_t size() const override;

    RawDataContainer_Status addElement(const void *element, size_t id) override;

    const char *getElement(size_t id) const override;

    RawDataContainer_Status removeElement(size_t id) override;

    RawDataContainer_Status updateElement(size_t id, const void *element) override;
};
