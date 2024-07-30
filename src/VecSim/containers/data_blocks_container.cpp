#include "data_blocks_container.h"

DataBlocksContainer::DataBlocksContainer(size_t blockSize, size_t elementBytesCount,
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         unsigned char _alignment)
    : VecsimBaseObject(allocator), RawDataContainer(), element_bytes_count(elementBytesCount),
      len(0), blocks(allocator), block_size(blockSize), alignment(_alignment) {}

DataBlocksContainer::~DataBlocksContainer() = default;

size_t DataBlocksContainer::size() const { return len; }

RawDataContainer_Status DataBlocksContainer::appendElement(const void *element) {
    if (len % block_size == 0) {
        blocks.push_back(DataBlock(this->block_size, this->element_bytes_count, this->allocator,
                                   this->alignment));
    }
    blocks.back().addElement(element);
    len++;
    return RAW_DATA_CONTAINER_OK;
}

const char *DataBlocksContainer::getElement(size_t id) const {
    if (id >= len) {
        return nullptr;
    }
    return blocks.at(id / this->block_size).getElement(id % this->block_size);
}

RawDataContainer_Status DataBlocksContainer::removeElement(size_t id) {
    if (id >= len) {
        return RAW_DATA_CONTAINER_ID_NOT_EXIST;
    }
    if (id < len - 1) {
        return RAW_DATA_CONTAINER_ERR; // only the last element can be removed
    }
    blocks.back().popLastElement();
    if (blocks.back().getLength() == 0) {
        blocks.pop_back();
    }
    len--;
    return RAW_DATA_CONTAINER_OK;
}

RawDataContainer_Status DataBlocksContainer::updateElement(size_t id, const void *element) {
    if (id >= len) {
        return RAW_DATA_CONTAINER_ID_NOT_EXIST;
    }
    auto &block = blocks.at(id / this->block_size);
    block.updateElement(id % block_size, element); // update the relative index in the block
    return RAW_DATA_CONTAINER_OK;
}
