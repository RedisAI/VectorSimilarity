#include "data_blocks_container.h"
#include "VecSim/utils/serializer.h"
#include <cmath>

DataBlocksContainer::DataBlocksContainer(size_t blockSize, size_t elementBytesCount,
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         unsigned char _alignment)
    : RawDataContainer(), vecsim_stl::vector<DataBlock>(allocator), element_bytes_count(elementBytesCount),
      element_count(0), block_size(blockSize), alignment(_alignment), is_set(true) {}

DataBlocksContainer::DataBlocksContainer(std::shared_ptr<VecSimAllocator> allocator):
RawDataContainer(), vecsim_stl::vector<DataBlock>(allocator),element_bytes_count(0),
      element_count(0), block_size(0), alignment(0), is_set(false) {}
DataBlocksContainer::~DataBlocksContainer() = default;

void DataBlocksContainer::operator delete(DataBlocksContainer *ptr, std::destroying_delete_t) noexcept {
  using alloc_type = std::shared_ptr<VecSimAllocator>;
  auto alloc = alloc_type{ptr->getAllocator()};
  ptr->~DataBlocksContainer();
  alloc->free_allocation(ptr);
}

size_t DataBlocksContainer::size() const { return element_count; }

// size_t DataBlocksContainer::capacity() const { return blocks.capacity(); }

size_t DataBlocksContainer::blockSize() const { return block_size; }

size_t DataBlocksContainer::elementByteCount() const { return element_bytes_count; }

RawDataContainer::Status DataBlocksContainer::addElement(const void *element) {
    if (element_count % block_size == 0) {
        // TODO: need a unique name here...

        this->emplace_back();
    }
    this->back().addElement(element);
    element_count++;
    return Status::OK;
}

void DataBlocksContainer::emplace_back() {
    std::string block_file_name = "blk_" + std::to_string(vecsim_stl::vector<DataBlock>::size());
    vecsim_stl::vector<DataBlock>::emplace_back(block_file_name.c_str(), this->block_size, this->element_bytes_count, this->allocator,
                        this->alignment);
}

RawDataContainer::Status DataBlocksContainer::addElement(const void *element, size_t id) {
    assert(id == element_count); // we can only append new elements
    return addElement(element);
}

const char *DataBlocksContainer::getElement(size_t id) const {
    assert(id < element_count);
    return at(id / this->block_size).getElement(id % this->block_size);
}

RawDataContainer::Status DataBlocksContainer::removeElement() {
    this->back().popLastElement();
    if (this->back().getLength() == 0) {
        this->pop_back();
    }
    element_count--;
    return Status::OK;
}

RawDataContainer::Status DataBlocksContainer::removeElement(size_t id) {
    assert(id == element_count - 1); // only the last element can be removed
    return removeElement();
}

RawDataContainer::Status DataBlocksContainer::updateElement(size_t id, const void *element) {
    assert(id < element_count);
    auto &block = at(id / this->block_size);
    block.updateElement(id % block_size, element); // update the relative index in the block
    return Status::OK;
}

std::unique_ptr<RawDataContainer::Iterator> DataBlocksContainer::getIterator() const {
    return std::make_unique<DataBlocksContainer::Iterator>(*this);
}

#ifdef BUILD_TESTS
void DataBlocksContainer::saveVectorsData(std::ostream &output) const {
    // Save data blocks
    for (size_t i = 0; i < this->numBlocks(); i++) {
        auto &block = this->at(i);
        unsigned int block_len = block.getLength();
        for (size_t j = 0; j < block_len; j++) {
            output.write(block.getElement(j), this->element_bytes_count);
        }
    }
}

void DataBlocksContainer::restoreBlocks(std::istream &input, size_t num_vectors,
                                        Serializer::EncodingVersion version) {

    // Get number of blocks
    unsigned int num_blocks = 0;
    if (version == Serializer::EncodingVersion_V3) {
        // In V3, the number of blocks is serialized, so we need to read it from the file.
        Serializer::readBinaryPOD(input, num_blocks);
    } else {
        // Otherwise, calculate the number of blocks based on the number of vectors.
        num_blocks = std::ceil((float)num_vectors / this->block_size);
    }
    this->reserve(num_blocks);

    // Get data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        this->emplace_back();
        unsigned int block_len = 0;
        if (version == Serializer::EncodingVersion_V3) {
            // In V3, the length of each block is serialized, so we need to read it from the file.
            Serializer::readBinaryPOD(input, block_len);
        } else {
            size_t vectors_left = num_vectors - this->element_count;
            block_len = vectors_left > this->block_size ? this->block_size : vectors_left;
        }
        for (size_t j = 0; j < block_len; j++) {
            auto cur_vec = this->getAllocator()->allocate_unique(this->element_bytes_count);
            input.read(static_cast<char *>(cur_vec.get()),
                       (std::streamsize)this->element_bytes_count);
            this->back().addElement(cur_vec.get());
            this->element_count++;
        }
    }
}

void DataBlocksContainer::shrinkToFit() { this->shrink_to_fit(); }

size_t DataBlocksContainer::numBlocks() const { return this->size(); }

#endif
/********************************** Iterator API ************************************************/

DataBlocksContainer::Iterator::Iterator(const DataBlocksContainer &container_)
    : RawDataContainer::Iterator(), cur_id(0), cur_element(nullptr), container(container_) {}

bool DataBlocksContainer::Iterator::hasNext() const {
    return this->cur_id != this->container.size();
}

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
