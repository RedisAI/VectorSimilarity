/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/spaces/computer/calculator.h"
#include "unit_test_utils.h"
#include "tests_utils.h"

class IndexCalculatorTest : public ::testing::Test {};
namespace dummyCalcultor {

using DummyType = int;
using dummy_dist_func_t = DummyType (*)(int);

int dummyDistFunc(int value) { return value; }

template <typename DistType>
class DistanceCalculatorDummy : public DistanceCalculatorInterface<DistType, dummy_dist_func_t> {
public:
    DistanceCalculatorDummy(std::shared_ptr<VecSimAllocator> allocator, dummy_dist_func_t dist_func)
        : DistanceCalculatorInterface<DistType, dummy_dist_func_t>(allocator, dist_func) {}

    virtual DistType calcDistance(const void *v1, const void *v2, size_t dim) const {
        return this->dist_func(7);
    }
};

} // namespace dummyCalcultor

TEST(IndexCalculatorTest, TestIndexCalculator) {

    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    // Test computer with a distance function signature different from dim(v1, v2, dim()).
    using namespace dummyCalcultor;
    auto distance_calculator = DistanceCalculatorDummy<DummyType>(allocator, dummyDistFunc);

    ASSERT_EQ(distance_calculator.calcDistance(nullptr, nullptr, 0), 7);
}

class PreprocessorsTest : public ::testing::Test {};

namespace dummyPreprocessors {

using DummyType = int;

enum pp_mode { STORAGE_ONLY, QUERY_ONLY, BOTH, EMPTY };

// Dummy storage preprocessor
template <typename DataType>
class DummyStoragePreprocessor : public PreprocessorInterface {
public:
    DummyStoragePreprocessor(std::shared_ptr<VecSimAllocator> allocator,
                             DataType value_to_add_storage, DataType value_to_add_query = 0)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(value_to_add_query) {
        if (!value_to_add_query)
            value_to_add_query = value_to_add_storage;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {

        this->preprocessForStorage(original_blob, storage_blob, input_blob_size);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate(input_blob_size);
            memcpy(blob, original_blob, input_blob_size);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_storage;
    }
    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {}

    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {
        assert(blob);
        static_cast<DataType *>(blob)[0] += value_to_add_storage;
    }

    void preprocessQuery(const void *original_blob, void *&blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        /* do nothing*/
    }

private:
    DataType value_to_add_storage;
    DataType value_to_add_query;
};

// Dummy query preprocessor
template <typename DataType>
class DummyQueryPreprocessor : public PreprocessorInterface {
public:
    DummyQueryPreprocessor(std::shared_ptr<VecSimAllocator> allocator, int value_to_add_storage,
                           int _value_to_add_query = 0)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(_value_to_add_query) {
        if (!_value_to_add_query)
            value_to_add_query = value_to_add_storage;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        this->preprocessQuery(original_blob, query_blob, input_blob_size, alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        /* do nothing*/
    }
    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {
        static_cast<DataType *>(blob)[0] += value_to_add_query;
    }
    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {}
    void preprocessQuery(const void *original_blob, void *&blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(input_blob_size, alignment);
            memcpy(blob, original_blob, input_blob_size);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_query;
    }

private:
    int value_to_add_storage;
    int value_to_add_query;
};

// Dummy mixed preprocessor (precesses the blobs  differently)
template <typename DataType>
class DummyMixedPreprocessor : public PreprocessorInterface {
public:
    DummyMixedPreprocessor(std::shared_ptr<VecSimAllocator> allocator, int value_to_add_storage,
                           int value_to_add_query)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(value_to_add_query) {}
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {

        // One blob was already allocated by a previous preprocessor(s) that process both blobs the
        // same. The blobs are pointing to the same memory, we need to allocate another memory slot
        // to split them.
        if ((storage_blob == query_blob) && (query_blob != nullptr)) {
            storage_blob = this->allocator->allocate(input_blob_size);
            memcpy(storage_blob, query_blob, input_blob_size);
        }

        // Either both are nullptr or they are pointing to different memory slots. Both cases are
        // handled by the designated functions.
        this->preprocessForStorage(original_blob, storage_blob, input_blob_size);
        this->preprocessQuery(original_blob, query_blob, input_blob_size, alignment);
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate(input_blob_size);
            memcpy(blob, original_blob, input_blob_size);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_storage;
    }
    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {}

    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {}
    void preprocessQuery(const void *original_blob, void *&blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        // If the blob was not allocated yet, allocate it.
        if (blob == nullptr) {
            blob = this->allocator->allocate_aligned(input_blob_size, alignment);
            memcpy(blob, original_blob, input_blob_size);
        }
        static_cast<DataType *>(blob)[0] += value_to_add_query;
    }

private:
    int value_to_add_storage;
    int value_to_add_query;
};

// TODO: test increase allocation size ( we don't really need another pp class for this)
// A preprocessor that changes the allocation size of the blobs in the same manner.
// set excess bytes to (char)2
template <typename DataType>
class DummyChangeAllocSizePreprocessor : public PreprocessorInterface {
private:
    size_t processed_bytes_count;
    static constexpr unsigned char excess_value = 2;

public:
    DummyChangeAllocSizePreprocessor(std::shared_ptr<VecSimAllocator> allocator,
                                     size_t processed_bytes_count)
        : PreprocessorInterface(allocator), processed_bytes_count(processed_bytes_count) {}

    static constexpr unsigned char getExcessValue() { return excess_value; }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        // if the blobs are equal,
        if (storage_blob == query_blob) {
            preprocessGeneral(original_blob, storage_blob, input_blob_size, alignment);
            query_blob = storage_blob;
            return;
        }
        // The blobs are not equal

        // If the input blob size is not enough
        if (input_blob_size < processed_bytes_count) {
            auto alloc_and_process = [&](void *&blob) {
                auto new_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                if (blob == nullptr) {
                    memcpy(new_blob, original_blob, input_blob_size);
                } else {
                    // copy the blob to the new blob, and free the old one
                    memcpy(new_blob, blob, input_blob_size);
                    this->allocator->free_allocation(blob);
                }
                blob = new_blob;
                memset((char *)blob + input_blob_size, excess_value,
                       processed_bytes_count - input_blob_size);
            };

            alloc_and_process(storage_blob);
            alloc_and_process(query_blob);
        } else {
            auto alloc_and_process = [&](void *&blob) {
                if (blob == nullptr) {
                    blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                    memcpy(blob, original_blob, processed_bytes_count);
                } else {
                    memset((char *)blob + processed_bytes_count, excess_value,
                           input_blob_size - processed_bytes_count);
                }
            };

            alloc_and_process(storage_blob);
            alloc_and_process(query_blob);
        }

        // update the input blob size
        input_blob_size = processed_bytes_count;
    }

    void preprocessForStorage(const void *original_blob, void *&blob,
                              size_t &input_blob_size) const override {

        this->preprocessGeneral(original_blob, blob, input_blob_size);
    }
    void preprocessQueryInPlace(void *blob, size_t input_blob_size,
                                unsigned char alignment) const override {
        // only supported if the blob in already large enough
        assert(input_blob_size >= processed_bytes_count);
        // set excess bytes to 0
        memset((char *)blob + processed_bytes_count, excess_value,
               input_blob_size - processed_bytes_count);
    }

    void preprocessStorageInPlace(void *blob, size_t input_blob_size) const override {
        // only supported if the blob in already large enough
        assert(input_blob_size >= processed_bytes_count);
        // set excess bytes to 0
        memset((char *)blob + processed_bytes_count, excess_value,
               input_blob_size - processed_bytes_count);
    }
    void preprocessQuery(const void *original_blob, void *&blob, size_t &input_blob_size,
                         unsigned char alignment) const override {
        this->preprocessGeneral(original_blob, blob, input_blob_size, alignment);
    }

private:
    void preprocessGeneral(const void *original_blob, void *&blob, size_t &input_blob_size,
                           unsigned char alignment = 0) const {
        // if the size of the input is not enough.
        if (input_blob_size < processed_bytes_count) {
            // calloc doesn't have an alignment version, so we need to allocate aligned memory and
            // cset the excess bytes to 0.
            auto new_blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
            if (blob == nullptr) {
                // copy thr original blob
                memcpy(new_blob, original_blob, input_blob_size);
            } else {
                // copy the blob to the new blob
                memcpy(new_blob, blob, input_blob_size);
                this->allocator->free_allocation(blob);
            }
            blob = new_blob;
            // set excess bytes to 0
            memset((char *)blob + input_blob_size, excess_value,
                   processed_bytes_count - input_blob_size);
        } else { // input size is larger than output
            if (blob == nullptr) {
                blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                memcpy(blob, original_blob, processed_bytes_count);
            } else {
                // set excess bytes to 0
                memset((char *)blob + processed_bytes_count, excess_value,
                       input_blob_size - processed_bytes_count);
            }
        }
        // update the input blob size
        input_blob_size = processed_bytes_count;
    }
};
} // namespace dummyPreprocessors

TEST(PreprocessorsTest, PreprocessorsTestBasicAlignmentTest) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    unsigned char alignment = 5;
    auto preprocessor = PreprocessorsContainerAbstract(allocator, alignment);
    const int original_blob[4] = {1, 1, 1, 1};
    size_t processed_bytes_count = sizeof(original_blob);

    {
        auto aligned_query = preprocessor.preprocessQuery(original_blob, processed_bytes_count);
        unsigned char address_alignment = (uintptr_t)(aligned_query.get()) % alignment;
        ASSERT_EQ(address_alignment, 0);
    }
}

template <unsigned char alignment>
void MultiPPContainerEmpty() {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t dim = 4;
    const int original_blob[dim] = {1, 2, 3, 4};
    const int original_blob_cpy[dim] = {1, 2, 3, 4};

    constexpr size_t n_preprocessors = 3;

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        // Original blob should not be changed
        CompareVectors(original_blob, original_blob_cpy, dim);

        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();

        // Storage blob should not be reallocated or changed
        ASSERT_EQ(storage_blob, (const int *)original_blob);
        CompareVectors(original_blob, (const int *)storage_blob, dim);

        // query blob *values* should not be changed
        CompareVectors(original_blob, (const int *)query_blob, dim);

        // If alignment is set the query blob address should be aligned to the specified alignment.
        if constexpr (alignment) {
            unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
            ASSERT_EQ(address_alignment, 0);
        }
    }
}

TEST(PreprocessorsTest, MultiPPContainerEmptyNoAlignment) {
    using namespace dummyPreprocessors;
    MultiPPContainerEmpty<0>();
}

TEST(PreprocessorsTest, MultiPPContainerEmptyAlignment) {
    using namespace dummyPreprocessors;
    MultiPPContainerEmpty<5>();
}

template <typename PreprocessorType>
void MultiPreprocessorsContainerNoAlignment(dummyPreprocessors::pp_mode MODE) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    unsigned char alignment = 0;
    int initial_value = 1;
    int value_to_add = 7;
    const int original_blob[4] = {initial_value, initial_value, initial_value, initial_value};
    size_t processed_bytes_count = sizeof(original_blob);

    // Test computer with multiple preprocessors of the same type.
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    auto verify_preprocess = [&](int expected_processed_value) {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);
        // Original blob should not be changed
        ASSERT_EQ(original_blob[0], initial_value);

        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        if (MODE == STORAGE_ONLY) {
            // New storage blob should be allocated
            ASSERT_NE(storage_blob, original_blob);
            // query blob should be unprocessed
            ASSERT_EQ(query_blob, original_blob);
            ASSERT_EQ(((const int *)storage_blob)[0], expected_processed_value);
        } else if (MODE == QUERY_ONLY) {
            // New query blob should be allocated
            ASSERT_NE(query_blob, original_blob);
            // Storage blob should be unprocessed
            ASSERT_EQ(storage_blob, original_blob);
            ASSERT_EQ(((const int *)query_blob)[0], expected_processed_value);
        }
    };

    /* ==== Add the first preprocessor ==== */
    auto preprocessor0 = new (allocator) PreprocessorType(allocator, value_to_add);
    // add preprocessor returns next free spot in its preprocessors array.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor0), 1);
    verify_preprocess(initial_value + value_to_add);

    /* ==== Add the second preprocessor ==== */
    auto preprocessor1 = new (allocator) PreprocessorType(allocator, value_to_add);
    // add preprocessor returns 0 when adding the last preprocessor.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor1), 0);
    ASSERT_NO_FATAL_FAILURE(verify_preprocess(initial_value + 2 * value_to_add));
}

TEST(PreprocessorsTest, MultiPreprocessorsContainerStorageNoAlignment) {
    using namespace dummyPreprocessors;
    MultiPreprocessorsContainerNoAlignment<DummyStoragePreprocessor<DummyType>>(
        pp_mode::STORAGE_ONLY);
}

TEST(PreprocessorsTest, MultiPreprocessorsContainerQueryNoAlignment) {
    using namespace dummyPreprocessors;
    MultiPreprocessorsContainerNoAlignment<DummyQueryPreprocessor<DummyType>>(pp_mode::QUERY_ONLY);
}

template <typename FirstPreprocessorType, typename SecondPreprocessorType>
void multiPPContainerMixedPreprocessorNoAlignment() {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 3;
    unsigned char alignment = 0;
    int initial_value = 1;
    int value_to_add_storage = 7;
    int value_to_add_query = 2;
    const int original_blob[4] = {initial_value, initial_value, initial_value, initial_value};
    size_t processed_bytes_count = sizeof(original_blob);

    // Test multiple preprocessors of the same type.
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    /* ==== Add one preprocessor of each type ==== */
    auto preprocessor0 =
        new (allocator) FirstPreprocessorType(allocator, value_to_add_storage, value_to_add_query);
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor0), 1);
    auto preprocessor1 =
        new (allocator) SecondPreprocessorType(allocator, value_to_add_storage, value_to_add_query);
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor1), 2);

    // scope this section so the blobs are released before the allocator.
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);
        // Original blob should not be changed
        ASSERT_EQ(original_blob[0], initial_value);

        // Both blobs should be allocated
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();

        // Ensure the computer process returns a new allocation of the expected processed blob with
        // the new value.
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
        ASSERT_NE(query_blob, storage_blob);

        ASSERT_EQ(((const int *)storage_blob)[0], initial_value + value_to_add_storage);
        ASSERT_EQ(((const int *)query_blob)[0], initial_value + value_to_add_query);
    }

    /* ==== Add a preprocessor that processes both storage and query ==== */
    auto preprocessor2 = new (allocator)
        DummyMixedPreprocessor<DummyType>(allocator, value_to_add_storage, value_to_add_query);
    // add preprocessor returns 0 when adding the last preprocessor.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor2), 0);
    {
        ProcessedBlobs mixed_processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);

        const void *mixed_pp_storage_blob = mixed_processed_blobs.getStorageBlob();
        const void *mixed_pp_query_blob = mixed_processed_blobs.getQueryBlob();

        // Ensure the computer process both blobs.
        ASSERT_EQ(((const int *)mixed_pp_storage_blob)[0],
                  initial_value + 2 * value_to_add_storage);
        ASSERT_EQ(((const int *)mixed_pp_query_blob)[0], initial_value + 2 * value_to_add_query);
    }

    // try adding another preprocessor and fail.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor2), -1);
}

TEST(PreprocessorsTest, multiPPContainerMixedPreprocessorQueryFirst) {
    using namespace dummyPreprocessors;
    multiPPContainerMixedPreprocessorNoAlignment<DummyQueryPreprocessor<DummyType>,
                                                 DummyStoragePreprocessor<DummyType>>();
}

TEST(PreprocessorsTest, multiPPContainerMixedPreprocessorStorageFirst) {
    using namespace dummyPreprocessors;
    multiPPContainerMixedPreprocessorNoAlignment<DummyStoragePreprocessor<DummyType>,
                                                 DummyQueryPreprocessor<DummyType>>();
}

template <typename PreprocessorType>
void multiPPContainerAlignment(dummyPreprocessors::pp_mode MODE) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    unsigned char alignment = 5;
    constexpr size_t n_preprocessors = 1;
    int initial_value = 1;
    int value_to_add = 7;
    const int original_blob[4] = {initial_value, initial_value, initial_value, initial_value};
    size_t processed_bytes_count = sizeof(original_blob);

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    auto verify_preprocess = [&](int expected_processed_value) {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, processed_bytes_count);

        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        if (MODE == STORAGE_ONLY) {
            // New storage blob should be allocated and processed
            ASSERT_NE(storage_blob, original_blob);
            ASSERT_EQ(((const int *)storage_blob)[0], expected_processed_value);
            // query blob *values* should be unprocessed, however, it might be allocated if the
            // original blob is not aligned.
            ASSERT_EQ(((const int *)query_blob)[0], original_blob[0]);
        } else if (MODE == QUERY_ONLY) {
            // New query blob should be allocated
            ASSERT_NE(query_blob, original_blob);
            // Storage blob should be unprocessed and not allocated.
            ASSERT_EQ(storage_blob, original_blob);
            ASSERT_EQ(((const int *)query_blob)[0], expected_processed_value);
        }

        // anyway the query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);
    };

    auto preprocessor0 = new (allocator) PreprocessorType(allocator, value_to_add);
    // add preprocessor returns next free spot in its preprocessors array.
    ASSERT_EQ(multiPPContainer.addPreprocessor(preprocessor0), 0);
    verify_preprocess(initial_value + value_to_add);
}

TEST(PreprocessorsTest, StoragePreprocessorWithAlignment) {
    using namespace dummyPreprocessors;
    multiPPContainerAlignment<DummyStoragePreprocessor<DummyType>>(pp_mode::STORAGE_ONLY);
}

TEST(PreprocessorsTest, QueryPreprocessorWithAlignment) {
    using namespace dummyPreprocessors;
    multiPPContainerAlignment<DummyQueryPreprocessor<DummyType>>(pp_mode::QUERY_ONLY);
}

TEST(PreprocessorsTest, multiPPContainerCosineThenMixedPreprocess) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t dim = 4;
    unsigned char alignment = 5;

    float initial_value = 1.0f;
    float normalized_value = 0.5f;
    float value_to_add_storage = 7.0f;
    float value_to_add_query = 2.0f;
    const float original_blob[dim] = {initial_value, initial_value, initial_value, initial_value};
    // TODo: change this test so that original_blob_size != processed_bytes_count
    constexpr size_t processed_bytes_count = sizeof(original_blob);

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    // adding cosine preprocessor
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<float>(allocator, dim, processed_bytes_count);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_EQ(storage_blob, query_blob);
        // memory should be aligned
        unsigned char address_alignment = (uintptr_t)(storage_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);
        // They need to be allocated and processed
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_EQ(((const float *)storage_blob)[0], normalized_value);
        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
    }
    // adding mixed preprocessor
    auto mixed_preprocessor = new (allocator)
        DummyMixedPreprocessor<float>(allocator, value_to_add_storage, value_to_add_query);
    multiPPContainer.addPreprocessor(mixed_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);

        // query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // They need to be processed by both processors.
        ASSERT_EQ(((const float *)storage_blob)[0], normalized_value + value_to_add_storage);
        ASSERT_EQ(((const float *)query_blob)[0], normalized_value + value_to_add_query);

        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }
    // The preprocessors should be released by the preprocessors container.
}

TEST(PreprocessorsTest, multiPPContainerMixedThenCosinePreprocess) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t dim = 4;
    unsigned char alignment = 5;

    float initial_value = 1.0f;
    float normalized_value = 0.5f;
    float value_to_add_storage = 7.0f;
    float value_to_add_query = 2.0f;
    const float original_blob[dim] = {initial_value, initial_value, initial_value, initial_value};
    constexpr size_t processed_bytes_count = sizeof(original_blob);

    // Creating multi preprocessors container
    auto mixed_preprocessor = new (allocator)
        DummyMixedPreprocessor<float>(allocator, value_to_add_storage, value_to_add_query);
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(mixed_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);

        // query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // They need to be processed by both processors.
        ASSERT_EQ(((const float *)storage_blob)[0], initial_value + value_to_add_storage);
        ASSERT_EQ(((const float *)query_blob)[0], initial_value + value_to_add_query);

        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }

    // adding cosine preprocessor
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<float>(allocator, dim, processed_bytes_count);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        // query memory should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);
        // They need to be allocated and processed
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);
        float expected_processed_storage[dim] = {initial_value + value_to_add_storage,
                                                 initial_value, initial_value, initial_value};
        float expected_processed_query[dim] = {initial_value + value_to_add_query, initial_value,
                                               initial_value, initial_value};
        VecSim_Normalize(expected_processed_storage, dim, VecSimType_FLOAT32);
        VecSim_Normalize(expected_processed_query, dim, VecSimType_FLOAT32);
        ASSERT_EQ(((const float *)storage_blob)[0], expected_processed_storage[0]);
        ASSERT_EQ(((const float *)query_blob)[0], expected_processed_query[0]);
        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }
    // The preprocessors should be released by the preprocessors container.
}
TEST(PreprocessorsTest, decrease_size_STORAGE_then_cosine_no_size_change) {}
TEST(PreprocessorsTest, decrease_size_QUERY_then_cosine_no_size_change) {}

TEST(PreprocessorsTest, DecreaseSizeThenFloatNormalize) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 8;
    constexpr size_t decrease_amount = 2;
    constexpr size_t new_elem_amount = elements - decrease_amount;

    // valgrind detects out of bound reads only if the considered memory is allocated on the heap,
    // rather than on the stack.
    constexpr size_t original_blob_size = elements * sizeof(float);
    auto original_blob_alloc = allocator->allocate_unique(original_blob_size);
    float *original_blob = static_cast<float *>(original_blob_alloc.get());
    test_utils::populate_float_vec(original_blob, elements);
    constexpr size_t new_processed_bytes_count =
        original_blob_size - decrease_amount * sizeof(float);

    // Processed blob expected output
    float expected_processed_blob[elements] = {0};
    memcpy(expected_processed_blob, original_blob, new_processed_bytes_count);
    // Only use half of the blob for normalization
    VecSim_Normalize(expected_processed_blob, new_elem_amount, VecSimType_FLOAT32);

    // A pp that decreases the original blob size
    auto decrease_size_preprocessor = new (allocator)
        DummyChangeAllocSizePreprocessor<float>(allocator, new_processed_bytes_count);
    // A normalize pp
    auto cosine_preprocessor = new (allocator)
        CosinePreprocessor<float>(allocator, new_elem_amount, new_processed_bytes_count);
    // Creating multi preprocessors container
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(decrease_size_preprocessor);
    multiPPContainer.addPreprocessor(cosine_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_EQ(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);

        // memory should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // They need to be allocated and processed
        EXPECT_NO_FATAL_FAILURE(CompareVectors<float>(static_cast<const float *>(storage_blob),
                                                      expected_processed_blob, new_elem_amount));
    }
}

TEST(PreprocessorsTest, Int8NormalizeThenIncreaseSize) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 8;

    // valgrind detects out of bound reads only if the considered memory is allocated on the heap,
    // rather than on the stack.
    constexpr size_t original_blob_size = elements * sizeof(int8_t);
    auto original_blob_alloc = allocator->allocate_unique(original_blob_size);
    int8_t *original_blob = static_cast<int8_t *>(original_blob_alloc.get());
    test_utils::populate_int8_vec(original_blob, elements);
    // size after normalization
    constexpr size_t normalized_blob_bytes_count = original_blob_size + sizeof(float);
    // size after increasing pp
    constexpr size_t elements_addition = 3;
    constexpr size_t final_blob_bytes_count =
        normalized_blob_bytes_count + elements_addition * sizeof(unsigned char);

    // Processed blob expected output
    int8_t expected_processed_blob[elements + sizeof(float) + 3] = {0};
    memcpy(expected_processed_blob, original_blob, original_blob_size);
    // normalize the original blob
    VecSim_Normalize(expected_processed_blob, elements, VecSimType_INT8);
    // add the values of the pp that increases the blob size
    unsigned char added_value = DummyChangeAllocSizePreprocessor<int8_t>::getExcessValue();
    for (size_t i = 0; i < elements_addition; i++) {
        expected_processed_blob[elements + sizeof(float) + i] = added_value;
    }

    // A normalize pp - will allocate the blob + sizeof(float)
    auto cosine_preprocessor = new (allocator)
        CosinePreprocessor<int8_t>(allocator, elements, normalized_blob_bytes_count);
    // A pp that will increase the *normalized* blob size
    auto increase_size_preprocessor =
        new (allocator) DummyChangeAllocSizePreprocessor<int8_t>(allocator, final_blob_bytes_count);
    // Creating multi preprocessors container
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    multiPPContainer.addPreprocessor(increase_size_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, sizeof(original_blob));
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_EQ(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);

        // memory should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // They need to be allocated and processed
        EXPECT_NO_FATAL_FAILURE(CompareVectors<int8_t>(static_cast<const int8_t *>(storage_blob),
                                                       expected_processed_blob,
                                                       final_blob_bytes_count));
    }
}

TEST(PreprocessorsTest, cosine_then_change_size) {
    // cosine (not changing)
    // pp that changes the blob size
}

TEST(PreprocessorsTest, cosine_change_then_pp_change) {
    // cosine ( changing)
    // pp that also changes the blob size
}

// TEST(PreprocessorsTest, multiPPContainerMixedThenCosinePreprocess) {
//     // add cosine pp that changes the original blob size
//     // add a pp that preprocesses the normalized blob (same size)
//     // add a pp that changes the storage_blob size, but not changing the query_blob size
// }

// TEST(PreprocessorsTest, multiPPContainerMixedThenCosinePreprocess) {
//     // add a pp that changes the storage_blob size, but not changing the query_blob size
//     // add a pp that preprocesses the normalized blob (same size)
//     // add cosine pp that changes the original blob size
// }

// TEST(PreprocessorsTest, multiPPContainerMixedThenCosinePreprocess) {
//     // pp multi container where cosine is only needed for the query blob (not supported yet)
// }
