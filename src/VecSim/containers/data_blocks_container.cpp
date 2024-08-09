#include "data_blocks_container.h"

DataBlocksContainer::DataBlocksContainer(size_t blockSize, size_t elementBytesCount,
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         unsigned char _alignment)
    : VecsimBaseObject(allocator), RawDataContainer(), element_bytes_count(elementBytesCount),
      element_count(0), blocks(allocator), block_size(blockSize), alignment(_alignment) {}

DataBlocksContainer::~DataBlocksContainer() = default;

size_t DataBlocksContainer::size() const { return element_count; }

size_t DataBlocksContainer::blockSize() const { return block_size; }

size_t DataBlocksContainer::elementByteCount() const { return element_bytes_count; }

RawDataContainer_Status DataBlocksContainer::addElement(const void *element, size_t id) {
    assert(id == element_count); // we can only append new elements
    if (element_count % block_size == 0) {
        blocks.emplace_back(this->block_size, this->element_bytes_count, this->allocator,
                            this->alignment);
    }
    blocks.back().addElement(element);
    element_count++;
    return RAW_DATA_CONTAINER_OK;
}

const char *DataBlocksContainer::getElement(size_t id) const {
    assert(id < element_count);
    return blocks.at(id / this->block_size).getElement(id % this->block_size);
}

RawDataContainer_Status DataBlocksContainer::removeElement(size_t id) {
    assert(id == element_count - 1); // only the last element can be removed
    blocks.back().popLastElement();
    if (blocks.back().getLength() == 0) {
        blocks.pop_back();
    }
    element_count--;
    return RAW_DATA_CONTAINER_OK;
}

RawDataContainer_Status DataBlocksContainer::updateElement(size_t id, const void *element) {
    assert(id < element_count);
    auto &block = blocks.at(id / this->block_size);
    block.updateElement(id % block_size, element); // update the relative index in the block
    return RAW_DATA_CONTAINER_OK;
}

std::unique_ptr<RawDataContainer::Iterator> DataBlocksContainer::getIterator() {
    return std::make_unique<DataBlocksContainer::Iterator>(*this);
}

/********************************** Iterator API ************************************************/

DataBlocksContainer::Iterator::Iterator(const DataBlocksContainer &container_)
    : RawDataContainer::Iterator(), cur_id(0), cur_element(nullptr), container(container_) {}

bool DataBlocksContainer::Iterator::hasNext() { return this->cur_id != this->container.size(); }

const char *DataBlocksContainer::Iterator::next() {
    if (!this->hasNext()) {
        return nullptr;
    }
    // Advance the pointer to the next element in the current block, or in the next block.
    if (this->cur_id % container.blockSize() == 0) {
        this->cur_element = container.getElement(this->cur_id);
    } else {
        this->cur_element += container.elementByteCount();
    }
    this->cur_id++;
    return this->cur_element;
}

void DataBlocksContainer::Iterator::reset() {
    this->cur_id = 0;
    this->cur_element = nullptr;
}
