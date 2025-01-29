#pragma once

#include "data_block.h"
#include "raw_data_container_interface.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/utils/vecsim_stl.h"

class DataBlocksContainer : public RawDataContainer, public vecsim_stl::vector<DataBlock> {

    size_t element_bytes_count;           // Element size in bytes
    size_t element_count;                 // Number of items in the container
    //  blocks; // data blocks
    size_t block_size;                    // number of element in block
    unsigned char alignment;              // alignment for data allocation in each block
    bool is_set;                            // Indicate wether the object parameters are set to ensure only set once.

public:
    DataBlocksContainer(size_t blockSize, size_t elementBytesCount,
                        std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment = 0);
    DataBlocksContainer(std::shared_ptr<VecSimAllocator> allocator);
    ~DataBlocksContainer();
    void operator delete(DataBlocksContainer *ptr, std::destroying_delete_t) noexcept;

    void setParams(size_t blockSize, size_t elementBytesCount, unsigned char alignment = 0) {
        if (is_set) {
            throw std::runtime_error("DataBlocksContainer parameters are already set");
        }
        element_bytes_count = elementBytesCount;
        block_size = blockSize;
        alignment = alignment;
        is_set = true;
    }

    size_t size() const override;

    // size_t capacity() const;

    size_t blockSize() const;

    size_t elementByteCount() const;

    Status addElement(const void *element, size_t id) override;
    Status addElement(const void *element);

    const char *getElement(size_t id) const override;

    Status removeElement(size_t id) override;
    Status removeElement();
    void emplace_back();

    Status updateElement(size_t id, const void *element) override;

    std::unique_ptr<RawDataContainer::Iterator> getIterator() const override;

#ifdef BUILD_TESTS
    void saveVectorsData(std::ostream &output) const override;
    // Use that in deserialization when file was created with old version (v3) that serialized
    // the blocks themselves and not just thw raw vector data.
    void restoreBlocks(std::istream &input, size_t num_vectors, Serializer::EncodingVersion);
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
