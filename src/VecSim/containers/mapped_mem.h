#pragma once
#include <sys/param.h>
#include <fcntl.h>
#include <sys/mman.h>
struct MappedMem {
    MappedMem() : mapped_addr(nullptr), curr_size(0) {
        // create a temporary file
        fd = open(".", O_TMPFILE | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
        if (fd == -1) {
            throw std::runtime_error("Failed to open file " + std::string("with error: ") +
                                     std::strerror(errno));
        }
    }

    void destroy(size_t element_size_bytes, size_t block_size_bytes) {
        if (!curr_size)
            return;
        // unmap memory
        size_t total_bytes = curr_size * element_size_bytes;
        size_t num_blocks = (total_bytes + block_size_bytes - 1) / block_size_bytes;
        size_t fileSize = num_blocks * block_size_bytes;
        munmap(mapped_addr, fileSize);
    }

    ~MappedMem() {
        // close file
        close(fd);
    }

    void appendElement(const void *element, size_t element_size_bytes) {
        // write element to the end of the file
        memcpy(mapped_addr + curr_size * element_size_bytes, element, element_size_bytes);
        ++curr_size;
    }

    const char *getElement(idType id, size_t element_size_bytes) const {
        return mapped_addr + id * element_size_bytes;
    }

    size_t get_elem_count() const { return curr_size; }

    bool is_full(size_t element_size_bytes, size_t block_size_bytes) const {
        // if curr_size * element_size_bytes is a multiple of block_size_bytes, return true
        return (curr_size * element_size_bytes) % block_size_bytes == 0;
    }

    // return true if the memory is full and we had to resize it
    bool growByBlock(size_t element_size_bytes, size_t block_size_bytes) {
        // if curr_size * element_size_bytes is a multiple of block_size_bytes, return true
        if (is_full(element_size_bytes, block_size_bytes)) {
            // Resize the file to the required size
            size_t curr_file_size_bytes = element_size_bytes * curr_size;
            size_t new_file_size = curr_file_size_bytes + block_size_bytes;
            if (posix_fallocate(fd, 0, new_file_size) < 0) {
                throw std::runtime_error("Failed to resize file " + std::string("with error: ") +
                                         std::strerror(errno));
            }

            if (curr_size) {
                char *remmapd_addr = (char *)mremap(mapped_addr, curr_file_size_bytes,
                                                    new_file_size, MREMAP_MAYMOVE);
                if (remmapd_addr == MAP_FAILED) {
                    throw std::runtime_error("Failed to remmap memory " +
                                             std::string("with error: ") + std::strerror(errno));
                }
                mapped_addr = remmapd_addr;
            } else { // first initialization
                int mmap_flags = MAP_PRIVATE;

                // map memory
                mapped_addr = static_cast<char *>(
                    mmap(NULL, new_file_size, PROT_READ | PROT_WRITE, mmap_flags, fd, 0));
                if (mapped_addr == MAP_FAILED) {
                    throw std::runtime_error("Failed to map file " + std::string("with error: ") +
                                             std::strerror(errno));
                }
            }
            // Give advise about sequential access to ensure the entire element is loaded into
            // memory
            // TODO: benchmark different madvise options
            if (madvise(mapped_addr, new_file_size, MADV_SEQUENTIAL) == -1) {
                throw std::runtime_error("madvise failed " + std::string("with error: ") +
                                         std::strerror(errno));
            }
            return true;
        }

        return false;
    }

    char *mapped_addr;
    size_t curr_size; // current element count in mapped memory
    int fd;
};

struct VectorsMappedMemContainer : public VecsimBaseObject, public MappedMem {

    VectorsMappedMemContainer(size_t block_size_bytes, size_t element_size_bytes,
                              std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), MappedMem(), element_bytes_count(element_size_bytes),
          block_size_bytes(block_size_bytes) {}

    const char *getElement(size_t id) const { return MappedMem::getElement(id, element_bytes_count); }

    void addElement(const void *elem, size_t id) {
        assert(id == curr_size);
        // grow if needed
        growByBlock(element_bytes_count, block_size_bytes);
        this->appendElement(elem, element_bytes_count);
    }

    size_t element_bytes_count;
    size_t block_size_bytes;

    /************************ No op functions to enable compilation *******************/
    void removeElement(size_t id) {}
    void updateElement(size_t id, const void *element) {}

    struct Iterator {
        /**
         * This is an abstract interface, constructor/destructor should be implemented by the
         * derived classes
         */
        explicit Iterator(const VectorsMappedMemContainer& container_): container(container_), cur_id(0){};
        virtual ~Iterator() = default;

        /**
         * The basic iterator operations API
         */
        virtual bool hasNext() const { return this->cur_id != this->container.curr_size; };
        virtual const char *next() {
            if (this->hasNext()) {
                return this->container.getElement(this->cur_id++);
            }
            return nullptr;

        }
        virtual void reset() {cur_id = 0;};

        const VectorsMappedMemContainer &container;
        size_t cur_id;
    };

    /**
     * Create a new iterator. Should be freed by the iterator's destroctor.
     */
    std::unique_ptr<Iterator> getIterator() const {
        return std::make_unique<VectorsMappedMemContainer::Iterator>(*this);
    }
};
