/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/computer/preprocessor_container.h"

/**
 *
=========================== TODO =================
original_blob_size should be a reference so it can be changed when changes in the pp chain.


*/
ProcessedBlobs PreprocessorsContainerAbstract::preprocess(const void *original_blob,
                                                          size_t original_blob_size) const {
    return ProcessedBlobs(preprocessForStorage(original_blob, original_blob_size),
                          preprocessQuery(original_blob, original_blob_size));
}

MemoryUtils::unique_blob
PreprocessorsContainerAbstract::preprocessForStorage(const void *original_blob,
                                                     size_t original_blob_size) const {
    return wrapWithDummyDeleter(const_cast<void *>(original_blob));
}

MemoryUtils::unique_blob PreprocessorsContainerAbstract::preprocessQuery(const void *original_blob,
                                                                         size_t original_blob_size,
                                                                         bool force_copy) const {
    return maybeCopyToAlignedMem(original_blob, original_blob_size, force_copy);
}

void PreprocessorsContainerAbstract::preprocessQueryInPlace(void *blob,
                                                            size_t input_blob_bytes_count) const {}

void PreprocessorsContainerAbstract::preprocessStorageInPlace(void *blob,
                                                              size_t input_blob_bytes_count) const {
}

MemoryUtils::unique_blob PreprocessorsContainerAbstract::maybeCopyToAlignedMem(
    const void *original_blob, size_t original_blob_size, bool force_copy) const {
    bool needs_copy =
        force_copy || (this->alignment && ((uintptr_t)original_blob % this->alignment != 0));

    if (needs_copy) {
        auto aligned_mem = this->allocator->allocate_aligned(original_blob_size, this->alignment);
        memcpy(aligned_mem, original_blob, original_blob_size);
        return this->wrapAllocated(aligned_mem);
    }

    // Returning a unique_ptr with a no-op deleter
    return wrapWithDummyDeleter(const_cast<void *>(original_blob));
}
