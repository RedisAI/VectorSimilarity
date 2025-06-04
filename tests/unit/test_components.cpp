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
#include "VecSim/spaces/computer/preprocessors.h"
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
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // This assert verifies that there's no use for this function for now - different sizes for
        // storage and query blobs. If such a use case arises, we can remove the assert and
        // implement the logic to handle different sizes.
        assert(storage_blob_size == query_blob_size);

        preprocess(original_blob, storage_blob, query_blob, storage_blob_size, alignment);
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
    DummyQueryPreprocessor(std::shared_ptr<VecSimAllocator> allocator,
                           DataType value_to_add_storage, DataType _value_to_add_query = 0)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(_value_to_add_query) {
        if (!_value_to_add_query)
            value_to_add_query = value_to_add_storage;
    }

    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // This assert verifies that there's no use for this function for now - different sizes for
        // storage and query blobs. If such a use case arises, we can remove the assert and
        // implement the logic to handle different sizes.
        assert(storage_blob_size == query_blob_size);

        preprocess(original_blob, storage_blob, query_blob, storage_blob_size, alignment);
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
    DataType value_to_add_storage;
    DataType value_to_add_query;
};

// Dummy mixed preprocessor (precesses the blobs  differently)
template <typename DataType>
class DummyMixedPreprocessor : public PreprocessorInterface {
public:
    DummyMixedPreprocessor(std::shared_ptr<VecSimAllocator> allocator,
                           DataType value_to_add_storage, DataType value_to_add_query)
        : PreprocessorInterface(allocator), value_to_add_storage(value_to_add_storage),
          value_to_add_query(value_to_add_query) {}
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        preprocess(original_blob, storage_blob, query_blob, storage_blob_size, alignment);
    }

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
    DataType value_to_add_storage;
    DataType value_to_add_query;
};

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
                    size_t &storage_blob_size, size_t &query_blob_size,
                    unsigned char alignment) const override {
        // if the blobs are equal,
        if (storage_blob == query_blob) {
            preprocessGeneral(original_blob, storage_blob, storage_blob_size, alignment);
            query_blob = storage_blob;
            query_blob_size = storage_blob_size;
        }
    }

    // If the input blob size is not enough
    void preprocess(const void *original_blob, void *&storage_blob, void *&query_blob,
                    size_t &input_blob_size, unsigned char alignment) const override {
        // if the blobs are equal,
        if (storage_blob == query_blob) {
            preprocessGeneral(original_blob, storage_blob, input_blob_size, alignment);
            query_blob = storage_blob;
            return;
        }
        // The blobs are not equal

        auto alloc_and_process = [&](void *&blob) {
            // If the input blob size is not enough
            if (input_blob_size < processed_bytes_count) {
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
            } else {
                if (blob == nullptr) {
                    blob = this->allocator->allocate_aligned(processed_bytes_count, alignment);
                    memcpy(blob, original_blob, processed_bytes_count);
                } else {
                    memset((char *)blob + processed_bytes_count, excess_value,
                           input_blob_size - processed_bytes_count);
                }
            }
        };

        alloc_and_process(storage_blob);
        alloc_and_process(query_blob);

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
    size_t original_blob_size = sizeof(original_blob);

    // Test computer with multiple preprocessors of the same type.
    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    auto verify_preprocess = [&](int expected_processed_value) {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
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
    size_t original_blob_size = sizeof(original_blob);

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
            multiPPContainer.preprocess(original_blob, original_blob_size);
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
            multiPPContainer.preprocess(original_blob, original_blob_size);

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
    size_t original_blob_size = sizeof(original_blob);

    auto multiPPContainer =
        MultiPreprocessorsContainer<DummyType, n_preprocessors>(allocator, alignment);

    auto verify_preprocess = [&](int expected_processed_value) {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);

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
    constexpr size_t original_blob_size = sizeof(original_blob);

    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);

    // adding cosine preprocessor
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<float>(allocator, dim, original_blob_size);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
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
            multiPPContainer.preprocess(original_blob, original_blob_size);
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

// Cosine pp receives different allocation for the storage and query blobs.
TEST(PreprocessorsTest, multiPPContainerMixedThenCosinePreprocess) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t dim = 4;
    unsigned char alignment = 5;

    // In this test the first preprocessor allocates the memory for both blobs, according to the
    // size passed by the pp container. The second preprocessor expects that if the blobs are
    // already allocated, their size matches its processed_bytes_count. Hence, the blob size should
    // be sufficient for the cosine preprocessing.
    constexpr int8_t normalized_blob_bytes_count = dim * sizeof(int8_t) + sizeof(float);
    auto blob_alloc = allocator->allocate_unique(normalized_blob_bytes_count);
    int8_t *original_blob = static_cast<int8_t *>(blob_alloc.get());
    test_utils::populate_int8_vec(original_blob, dim);

    // Processing params
    int8_t value_to_add_storage = 7;
    int8_t value_to_add_query = 4;
    // Creating multi preprocessors container
    auto mixed_preprocessor = new (allocator)
        DummyMixedPreprocessor<int8_t>(allocator, value_to_add_storage, value_to_add_query);
    auto multiPPContainer =
        MultiPreprocessorsContainer<int8_t, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(mixed_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, normalized_blob_bytes_count);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);

        // query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        // Both blobs were processed.
        ASSERT_EQ((static_cast<const int8_t *>(storage_blob))[0],
                  original_blob[0] + value_to_add_storage);
        ASSERT_EQ((static_cast<const int8_t *>(query_blob))[0],
                  original_blob[0] + value_to_add_query);

        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);
    }

    // Processed blob expected output
    auto expected_processed_blob = [&](int8_t *blob, int8_t value_to_add) {
        memcpy(blob, original_blob, normalized_blob_bytes_count);
        blob[0] += value_to_add;
        VecSim_Normalize(blob, dim, VecSimType_INT8);
    };

    int8_t expected_processed_storage[normalized_blob_bytes_count] = {0};
    expected_processed_blob(expected_processed_storage, value_to_add_storage);
    int8_t expected_processed_query[normalized_blob_bytes_count] = {0};
    expected_processed_blob(expected_processed_query, value_to_add_query);

    // normalize the original blob
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<int8_t>(allocator, dim, normalized_blob_bytes_count);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    {
        // An assertion should be raised by the cosine preprocessor for unmatching blob sizes.
        // in valgrind the test continues, but the assertion appears in its log looking like an
        // error, so to avoid confusion we skip this line in valgrind.
#if !defined(RUNNING_ON_VALGRIND) && !defined(NDEBUG)
        EXPECT_EXIT(
            {
                ProcessedBlobs processed_blobs = multiPPContainer.preprocess(
                    original_blob, normalized_blob_bytes_count - sizeof(float));
            },
            testing::KilledBySignal(SIGABRT), "input_blob_size == processed_bytes_count");
#endif
        // Use the correct size
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, normalized_blob_bytes_count);
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
        // the original blob should not change
        ASSERT_NE(storage_blob, original_blob);
        ASSERT_NE(query_blob, original_blob);

        EXPECT_NO_FATAL_FAILURE(CompareVectors<int8_t>(static_cast<const int8_t *>(storage_blob),
                                                       expected_processed_storage,
                                                       normalized_blob_bytes_count));
        EXPECT_NO_FATAL_FAILURE(CompareVectors<int8_t>(static_cast<const int8_t *>(query_blob),
                                                       expected_processed_query,
                                                       normalized_blob_bytes_count));
    }
    // The preprocessors should be released by the preprocessors container.
}

template <typename PreprocessorType>
void AsymmetricPPThenCosine(dummyPreprocessors::pp_mode MODE) {
    using namespace dummyPreprocessors;

    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t dim = 4;
    unsigned char alignment = 5;

    float original_blob[dim] = {0};
    constexpr size_t original_blob_size = dim * sizeof(float);
    test_utils::populate_float_vec(original_blob, dim);

    // Processing params
    float value_to_add_storage = 0;
    float value_to_add_query = 0;
    if (MODE == STORAGE_ONLY) {
        value_to_add_storage = 7;
    } else if (MODE == QUERY_ONLY) {
        value_to_add_query = 4;
    }

    // Processed blob expected output
    auto expected_processed_blob = [&](float *blob, float value_to_add) {
        memcpy(blob, original_blob, original_blob_size);
        blob[0] += value_to_add;
        VecSim_Normalize(blob, dim, VecSimType_FLOAT32);
    };

    float expected_processed_storage[dim] = {0};
    expected_processed_blob(expected_processed_storage, value_to_add_storage);
    float expected_processed_query[dim] = {0};
    expected_processed_blob(expected_processed_query, value_to_add_query);

    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
    // will allocate either the storage or the query blob, depending on the preprocessor type.
    // TODO : can we pass just value_to_add?
    auto asymmetric_preprocessor =
        new (allocator) PreprocessorType(allocator, value_to_add_storage, value_to_add_query);
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<float>(allocator, dim, original_blob_size);

    multiPPContainer.addPreprocessor(asymmetric_preprocessor);
    multiPPContainer.addPreprocessor(cosine_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to a different memory slot
        ASSERT_NE(storage_blob, query_blob);
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);

        // query blob should be aligned
        unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
        ASSERT_EQ(address_alignment, 0);

        EXPECT_NO_FATAL_FAILURE(CompareVectors<float>(static_cast<const float *>(storage_blob),
                                                      expected_processed_storage, dim));
        EXPECT_NO_FATAL_FAILURE(CompareVectors<float>(static_cast<const float *>(query_blob),
                                                      expected_processed_query, dim));
    }
}
// The first preprocessor in the chain allocates only the storage blob,
// and the responsibility of allocating the query blob is delegated to the next preprocessor
// in the chain (cosine preprocessor in this case).
TEST(PreprocessorsTest, STORAGE_then_cosine_no_size_change) {
    using namespace dummyPreprocessors;
    EXPECT_NO_FATAL_FAILURE(
        AsymmetricPPThenCosine<DummyStoragePreprocessor<float>>(pp_mode::STORAGE_ONLY));
}

TEST(PreprocessorsTest, QUERY_then_cosine_no_size_change) {
    using namespace dummyPreprocessors;
    EXPECT_NO_FATAL_FAILURE(
        AsymmetricPPThenCosine<DummyQueryPreprocessor<float>>(pp_mode::QUERY_ONLY));
}

// A test where the value of input_blob_size is modified by the first pp.
// The cosine pp processed_bytes_count should be set at initialization with the *modified* value,
// otherwise, an assertion will be raised by it for an allocated blob that is smaller than the
// expected size.
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
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
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

// In this test the original blob size is smaller than the processed_bytes_count of the
// cosine preprocessor. Before the bug fix, the cosine pp would try to copy processed_bytes_count
// bytes of the original blob to the allocated memory, which would cause an out of bound read,
// that should be detected by valgrind.
TEST(PreprocessorsTest, Int8NormalizeThenIncreaseSize) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t n_preprocessors = 2;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 7;

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
    int8_t expected_processed_blob[elements + sizeof(float) + elements_addition] = {0};
    memcpy(expected_processed_blob, original_blob, original_blob_size);
    // normalize the original blob
    VecSim_Normalize(expected_processed_blob, elements, VecSimType_INT8);
    // add the values of the pp that increases the blob size
    unsigned char added_value = DummyChangeAllocSizePreprocessor<int8_t>::getExcessValue();
    for (size_t i = 0; i < elements_addition; i++) {
        expected_processed_blob[elements + sizeof(float) + i] = static_cast<int8_t>(added_value);
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
        // Note: we pass the original blob size to detect out of bound reads in the cosine pp.
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
        EXPECT_NO_FATAL_FAILURE(CompareVectors<int8_t>(static_cast<const int8_t *>(storage_blob),
                                                       expected_processed_blob,
                                                       final_blob_bytes_count));
    }
}

// Tests the quantization preprocessor with a single preprocessor in the chain.
// The QuantPreprocessor allocates the storage blob and processes it, while the query blob
// is unprocessed and allocated by the preprocessors container.
TEST(PreprocessorsTest, QuantizationTest) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t n_preprocessors = 1;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 7;
    constexpr size_t original_blob_size = elements * sizeof(float);
    auto original_blob_alloc = allocator->allocate_unique(original_blob_size);
    float *original_blob = static_cast<float *>(original_blob_alloc.get());
    test_utils::populate_float_vec(original_blob, elements);

    auto quant_preprocessor = new (allocator) QuantPreprocessor(allocator, elements);
    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(quant_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(storage_blob, query_blob);

        constexpr size_t quantized_blob_bytes_count =
            elements * sizeof(uint8_t) + 2 * sizeof(float);
        auto expected_processed_blob_alloc = allocator->allocate_unique(quantized_blob_bytes_count);
        uint8_t *expected_processed_blob =
            static_cast<uint8_t *>(expected_processed_blob_alloc.get());

        quant_preprocessor->quantize(original_blob, expected_processed_blob);
        EXPECT_NO_FATAL_FAILURE(CompareVectors<uint8_t>(static_cast<const uint8_t *>(storage_blob),
                                                        expected_processed_blob,
                                                        quantized_blob_bytes_count));

        // Compare the min and delta values of the quantized blob and expected_processed_blob
        const float *storage_blob_metadata =
            reinterpret_cast<const float *>(static_cast<const uint8_t *>(storage_blob) + elements);
        const float *qexpected_processed_metadata =
            reinterpret_cast<const float *>(expected_processed_blob + elements);
        // Check that the min and delta values are close enough
        ASSERT_FLOAT_EQ(storage_blob_metadata[0], qexpected_processed_metadata[0]);
        ASSERT_FLOAT_EQ(storage_blob_metadata[1], qexpected_processed_metadata[1]);

        float min = storage_blob_metadata[0];
        float delta = storage_blob_metadata[1];
        // reconstruct the original blob from the quantized blob
        uint8_t *uint8_storage_blob = static_cast<uint8_t *>(const_cast<void *>(storage_blob));
        for (size_t i = 0; i < elements; i++) {
            float reconstructed_value = min + uint8_storage_blob[i] * delta;
            ASSERT_NEAR(reconstructed_value, original_blob[i], 0.01)
                << "Reconstructed blob differs from the original blob at index " << i;
        }
    }
}

// Tests the quantization preprocessor with a cosine preprocessor in the chain.
// The QuantPreprocessor receives an allocated blob from the cosine preprocessor.
TEST(PreprocessorsTest, QuantizationTestWithCosine) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t n_preprocessors = 2;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 7;
    constexpr size_t original_blob_size = elements * sizeof(float);
    auto original_blob_alloc = allocator->allocate_unique(original_blob_size);
    float *original_blob = static_cast<float *>(original_blob_alloc.get());
    test_utils::populate_float_vec(original_blob, elements);

    auto quant_preprocessor = new (allocator) QuantPreprocessor(allocator, elements);
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<float>(allocator, elements, original_blob_size);
    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    multiPPContainer.addPreprocessor(quant_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);
        ASSERT_NE(storage_blob, query_blob);

        auto expected_processed_blob_alloc = allocator->allocate_unique(original_blob_size);
        float *expected_processed_blob = static_cast<float *>(expected_processed_blob_alloc.get());
        memcpy(expected_processed_blob, original_blob, original_blob_size);
        VecSim_Normalize(expected_processed_blob, elements, VecSimType_FLOAT32);

        auto quantized_blob_alloc =
            allocator->allocate_unique(elements * sizeof(uint8_t) + 2 * sizeof(float));
        uint8_t *quantized_blob = static_cast<uint8_t *>(quantized_blob_alloc.get());

        // quantization should be applied after normalization
        quant_preprocessor->quantize(expected_processed_blob, quantized_blob);
        // compare the storage blob to the expected processed blob
        EXPECT_NO_FATAL_FAILURE(CompareVectors<uint8_t>(static_cast<const uint8_t *>(storage_blob),
                                                        quantized_blob, elements));
        const float *storage_blob_metadata =
            reinterpret_cast<const float *>(static_cast<const uint8_t *>(storage_blob) + elements);
        const float *qexpected_processed_metadata =
            reinterpret_cast<const float *>(quantized_blob + elements);
        // Check that the min and delta values are close enough
        ASSERT_FLOAT_EQ(storage_blob_metadata[0], qexpected_processed_metadata[0]);
        ASSERT_FLOAT_EQ(storage_blob_metadata[1], qexpected_processed_metadata[1]);
    }
}

// Tests the quantization preprocessor with a single preprocessor in the chain.
// The QuantPreprocessor allocates the storage blob with a size larger than the original blob.

TEST(PreprocessorsTest, ReallocateVectorQuantizationTest) {
    // Checks that if not enough memory was allocated by a previous preprocessor, the quantization
    // preprocessor will reallocate it.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t n_preprocessors = 2;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 2;
    constexpr size_t original_blob_size = elements * sizeof(float);
    auto original_blob_alloc = allocator->allocate_unique(original_blob_size);
    float *original_blob = static_cast<float *>(original_blob_alloc.get());
    test_utils::populate_float_vec(original_blob, elements);

    auto quant_preprocessor = new (allocator) QuantPreprocessor(allocator, elements);
    auto dummy_preprocessor =
        new (allocator) dummyPreprocessors::DummyStoragePreprocessor<float>(allocator, 0.0f);
    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(dummy_preprocessor);
    multiPPContainer.addPreprocessor(quant_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(storage_blob, query_blob);

        constexpr size_t quantized_blob_bytes_count =
            elements * sizeof(uint8_t) + 2 * sizeof(float);
        uint8_t expected_processed_blob[quantized_blob_bytes_count] = {0};
        quant_preprocessor->quantize(original_blob, expected_processed_blob);
        EXPECT_NO_FATAL_FAILURE(CompareVectors<uint8_t>(static_cast<const uint8_t *>(storage_blob),
                                                        expected_processed_blob,
                                                        quantized_blob_bytes_count));
    }
}

// Tests the quantization preprocessor with a cosine preprocessor in the chain.
// The QuantPreprocessor receives an allocated blob from the cosine preprocessor, and needs to
// reallocate it. because the original blob size is smaller than the processed_bytes_count of the
// cosine preprocessor.
TEST(PreprocessorsTest, ReallocateVectorCosineQuantizationTest) {
    // Checks that if not enough memory was allocated by a previous preprocessor, the quantization
    // preprocessor will reallocate it.
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t n_preprocessors = 2;
    constexpr size_t alignment = 5;
    constexpr size_t elements = 2;
    constexpr size_t original_blob_size = elements * sizeof(float);
    auto original_blob_alloc = allocator->allocate_unique(original_blob_size);
    float *original_blob = static_cast<float *>(original_blob_alloc.get());
    for (size_t i = 0; i < elements; i++) {
        original_blob[i] = static_cast<float>(i + 2.5f);
    }

    auto quant_preprocessor = new (allocator) QuantPreprocessor(allocator, elements);
    auto cosine_preprocessor =
        new (allocator) CosinePreprocessor<float>(allocator, elements, original_blob_size);
    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(cosine_preprocessor);
    multiPPContainer.addPreprocessor(quant_preprocessor);

    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_blob, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(query_blob, nullptr);
        ASSERT_NE(storage_blob, query_blob);

        float expected_processed_blob[elements] = {0};
        memcpy(expected_processed_blob, original_blob, original_blob_size);
        VecSim_Normalize(expected_processed_blob, elements, VecSimType_FLOAT32);
        uint8_t quantized_blob[elements * sizeof(uint8_t) + 2 * sizeof(float)] = {0};
        // quantization should be applied after normalization
        quant_preprocessor->quantize(expected_processed_blob, quantized_blob);
        // compare the storage blob to the expected processed blob
        EXPECT_NO_FATAL_FAILURE(CompareVectors<uint8_t>(static_cast<const uint8_t *>(storage_blob),
                                                        quantized_blob, elements));
    }
}

TEST(PreprocessorsTest, QuantizationInPlaceTest) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    constexpr size_t alignment = 5;
    constexpr size_t dim = 5;
    constexpr size_t n_preprocessors = 2;
    constexpr size_t original_blob_size = dim * sizeof(float);
    constexpr size_t storage_bytes_count = dim * sizeof(uint8_t) + 2 * sizeof(float);

    // Create a float array with known values
    float original_data[dim] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto quant_preprocessor = new (allocator) QuantPreprocessor(allocator, dim);
    auto dummy_preprocessor =
        new (allocator) dummyPreprocessors::DummyStoragePreprocessor<float>(allocator, 0.0f);
    auto multiPPContainer =
        MultiPreprocessorsContainer<float, n_preprocessors>(allocator, alignment);
    multiPPContainer.addPreprocessor(dummy_preprocessor);
    multiPPContainer.addPreprocessor(quant_preprocessor);
    {
        ProcessedBlobs processed_blobs =
            multiPPContainer.preprocess(original_data, original_blob_size);
        const void *storage_blob = processed_blobs.getStorageBlob();
        const void *query_blob = processed_blobs.getQueryBlob();
        // blobs should point to the same memory slot
        ASSERT_NE(storage_blob, nullptr);
        ASSERT_NE(storage_blob, query_blob);

        // Verify the quantization results
        const uint8_t *quantized = static_cast<const uint8_t *>(storage_blob);
        const float *metadata = reinterpret_cast<const float *>(quantized + dim);

        // Extract metadata
        float min_val = metadata[0];
        float delta = metadata[1];

        // Check min_val is correct (should be 1.0f)
        ASSERT_FLOAT_EQ(min_val, 1.0f);

        // Check delta is correct ((5.0f - 1.0f) / 255.0f)
        ASSERT_FLOAT_EQ(delta, 4.0f / 255.0f);

        // dequantize and verify the values
        for (size_t i = 0; i < dim; ++i) {
            float dequantized_value = min_val + quantized[i] * delta;
            ASSERT_NEAR(dequantized_value, original_data[i], 0.01);
        }
    }
}

// Test the backward compatibility of the preprocess method with a single input_blob_size parameter
TEST(PreprocessorsTest, PreprocessBackwardCompatibilityTest) {
    using namespace dummyPreprocessors;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    constexpr size_t dim = 4;
    unsigned char alignment = 5;
    float initial_value = 1.0f;
    const float original_blob[dim] = {initial_value, initial_value, initial_value, initial_value};
    size_t original_blob_size = sizeof(original_blob);

    // Create a preprocessor that modifies both storage and query blobs
    auto mixed_preprocessor = new (allocator) DummyMixedPreprocessor<float>(allocator, 7.0f, 2.0f);

    // Test the backward compatibility method (single input_blob_size)
    void *storage_blob = nullptr;
    void *query_blob = nullptr;
    size_t input_blob_size = original_blob_size;

    // Call the backward compatibility version of preprocess
    mixed_preprocessor->preprocess(original_blob, storage_blob, query_blob, input_blob_size,
                                   alignment);

    // Verify that both blobs were allocated and processed correctly
    ASSERT_NE(storage_blob, nullptr);
    ASSERT_NE(query_blob, nullptr);
    ASSERT_NE(storage_blob, query_blob);

    // Verify that the input_blob_size was updated correctly
    ASSERT_EQ(input_blob_size, original_blob_size);

    // Verify that the storage blob was processed correctly
    ASSERT_EQ(((const float *)storage_blob)[0], initial_value + 7.0f);

    // Verify that the query blob was processed correctly
    ASSERT_EQ(((const float *)query_blob)[0], initial_value + 2.0f);

    // Verify that the query blob is aligned
    unsigned char address_alignment = (uintptr_t)(query_blob) % alignment;
    ASSERT_EQ(address_alignment, 0);
}
