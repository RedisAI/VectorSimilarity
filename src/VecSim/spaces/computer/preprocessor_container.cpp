/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/spaces/computer/preprocessor_container.h"

ProcessedBlobs PreprocessorsContainerAbstract::preprocess(const void *original_blob,
                                                          size_t processed_bytes_count) const {
    return ProcessedBlobs(preprocessForStorage(original_blob, processed_bytes_count),
                          preprocessQuery(original_blob, processed_bytes_count));
}

MemoryUtils::unique_blob
PreprocessorsContainerAbstract::preprocessForStorage(const void *original_blob,
                                                     size_t processed_bytes_count) const {
    return wrapWithDummyDeleter(const_cast<void *>(original_blob));
}

MemoryUtils::unique_blob PreprocessorsContainerAbstract::preprocessQuery(
    const void *original_blob, size_t processed_bytes_count, bool force_copy) const {
    return maybeCopyToAlignedMem(original_blob, processed_bytes_count, force_copy);
}

void PreprocessorsContainerAbstract::preprocessQueryInPlace(void *blob,
                                                            size_t processed_bytes_count) const {}

void PreprocessorsContainerAbstract::preprocessStorageInPlace(void *blob,
                                                              size_t processed_bytes_count) const {}

MemoryUtils::unique_blob PreprocessorsContainerAbstract::maybeCopyToAlignedMem(
    const void *original_blob, size_t blob_bytes_count, bool force_copy) const {
    bool needs_copy =
        force_copy || (this->alignment && ((uintptr_t)original_blob % this->alignment != 0));

    if (needs_copy) {
        auto aligned_mem = this->allocator->allocate_aligned(blob_bytes_count, this->alignment);
        memcpy(aligned_mem, original_blob, blob_bytes_count);
        return this->wrapAllocated(aligned_mem);
    }

    // Returning a unique_ptr with a no-op deleter
    return wrapWithDummyDeleter(const_cast<void *>(original_blob));
}
