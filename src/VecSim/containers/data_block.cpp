/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "data_block.h"
#include "VecSim/memory/vecsim_malloc.h"
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>

#define TWO_MB 2 * 1024 * 1024
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
#endif

DataBlock::DataBlock(size_t blockSize, size_t elementBytesCount,
                     std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment)
    : VecsimBaseObject(allocator), element_bytes_count(elementBytesCount), length(0),
      data((char *)this->allocator->allocate_aligned(blockSize * elementBytesCount, alignment)) {}

DataBlock::DataBlock(const char *file_name, size_t blockSize, size_t elementBytesCount,
              std::shared_ptr<VecSimAllocator> allocator, unsigned char alignment)
    : VecsimBaseObject(allocator), element_bytes_count(elementBytesCount), length(0) {

      // Create a file for the data block
      int fd = open(".", O_TMPFILE | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
      if (fd == -1) {
          throw std::runtime_error("Failed to open file " + std::string(file_name) + std::string("with error: ") + std::strerror(errno));
      }

      size_t blockSizeBytes = element_bytes_count * blockSize;
      // Allocate the file size
      blockSizeBytes += alignment; // Add enough space for alignment.
      if (ftruncate(fd, blockSizeBytes) == -1) {
        throw std::runtime_error("Failed to ftruncate file " + std::string(file_name));
          close(fd);
      }

      // Map the file to memory
      int mmap_flags = MAP_PRIVATE;

      // Adjust the block size to be a multiple of the page size
      size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
      if (element_bytes_count > pageSize) {
        // use HUGE_PAGES
        mmap_flags |= MAP_HUGETLB;
        if (element_bytes_count > TWO_MB) {
          mmap_flags |= MAP_HUGE_1GB;
        } else {
          mmap_flags |= MAP_HUGE_2MB;
        }
      }

      void *mapped_mem = mmap(NULL, blockSizeBytes, PROT_READ | PROT_WRITE, mmap_flags, fd, 0);

      if (mapped_mem == MAP_FAILED) {
          throw std::runtime_error("mmap failed");
      }
      close(fd);

      size_t remainder = ((uintptr_t)mapped_mem) % alignment;
      unsigned char offset = alignment - remainder;
      data = static_cast<char *>(mapped_mem) + offset;

      // Give advise about sequential access to ensure the entire element is loaded into memory
      // TODO: benchmark different madvise options
      if (madvise(data, blockSizeBytes, MADV_SEQUENTIAL) == -1) {
          throw std::runtime_error("madvise failed");
      }

}

DataBlock::DataBlock(DataBlock &&other) noexcept
    : VecsimBaseObject(other.allocator), element_bytes_count(other.element_bytes_count),
      length(other.length), data(other.data) {
    other.data = nullptr; // take ownership of the data
}

DataBlock::~DataBlock() noexcept {
  // munmap
    // to_clean_mem = data - offset
  // delete the file
}

void DataBlock::addElement(const void *element) {

    // Copy element data and update block size.
    memcpy(this->data + (this->length * element_bytes_count), element, element_bytes_count);
    this->length++;
}

void DataBlock::updateElement(size_t index, const void *new_element) {
    char *destinaion = (char *)getElement(index);
    memcpy(destinaion, new_element, element_bytes_count);
}
