#pragma once

#include "data_block.h"
#include "raw_data_container_interface.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/utils/vecsim_stl.h"

class DataBlocksContainer : public VecsimBaseObject, public RawDataContainer {

    size_t element_bytes_count;           // Element size in bytes
    size_t element_count;                 // Number of items in the container
    vecsim_stl::vector<DataBlock> blocks; // data blocks
    size_t block_size;                    // number of element in block
    unsigned char alignment;              // alignment for data allocation in each block

public:
    DataBlocksContainer(size_t blockSize, size_t elementBytesCount,
                        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment = 0);
    ~DataBlocksContainer();

    size_t size() const override;

    size_t capacity() const;

    size_t blockSize() const;

    size_t elementByteCount() const;

    Status addElement(const void *element, size_t id) override;

    const char *getElement(size_t id) const override;

    Status removeElement(size_t id) override;

    Status updateElement(size_t id, const void *element) override;

    std::unique_ptr<RawDataContainer::Iterator> getIterator() const override;

#ifdef BUILD_TESTS
#ifdef SERIALIZE

    void saveVectorsData(std::ostream &output) const override;
    // Use that in deserialization when file was created with old version (v3) that serialized
    // the blocks themselves and not just thw raw vector data.
    void restoreBlocks(std::istream &input, size_t num_vectors, Serializer::EncodingVersion);
#endif
    void shrinkToFit();
    size_t numBlocks() const;
#endif

    class Iterator : public RawDataContainer::Iterator {
        size_t cur_id;
        const char *cur_element;
        const DataBlocksContainer &container;

    public:
        explicit Iterator(const DataBlocksContainer &container);
        ~Iterator() override = default;

        bool hasNext() const override;
        const char *next() override;
        void reset() override;
    };
};
