/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "data_block.h"
#include <cstring>

DataBlock::DataBlock(size_t blockSize, size_t elementBytesCount)
    : element_bytes_count(elementBytesCount), length(0), data(), buf((char*)malloc(elementBytesCount)) {
    data.open(std::tmpnam(nullptr), // Not thread/process safe
              std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
}

DataBlock::DataBlock(DataBlock &&other) noexcept
    : element_bytes_count(other.element_bytes_count), length(other.length), data(std::move(other.data)), buf(other.buf) {
    other.buf = nullptr;
}

DataBlock::~DataBlock() noexcept { free(buf); }

void DataBlock::addElement(const void *element) {
    // Copy element data and update block size.
    this->data.pubseekpos(this->length * element_bytes_count);
    this->data.sputn((const char *)element, element_bytes_count);
    this->length++;
}

void DataBlock::updateElement(size_t index, const void *new_element) {
    this->data.pubseekpos(index * element_bytes_count);
    this->data.sputn((const char *)new_element, element_bytes_count);
}
