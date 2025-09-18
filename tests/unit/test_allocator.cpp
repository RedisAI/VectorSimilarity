/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "test_utils.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"

const size_t vecsimAllocationOverhead = sizeof(size_t);

const size_t hashTableNodeSize = getLabelsLookupNodeSize();

class AllocatorTest : public ::testing::Test {};
struct SimpleObject : public VecsimBaseObject {
public:
    SimpleObject(std::shared_ptr<VecSimAllocator> allocator) : VecsimBaseObject(allocator) {}
    int x;
};

struct ObjectWithSTL : public VecsimBaseObject {
    std::vector<int, VecsimSTLAllocator<int>> test_vec;

public:
    ObjectWithSTL(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), test_vec(allocator){};
};

struct NestedObject : public VecsimBaseObject {
    ObjectWithSTL stl_object;
    SimpleObject simpleObject;

public:
    NestedObject(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), stl_object(allocator), simpleObject(allocator){};
};

TEST_F(AllocatorTest, test_simple_object) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    SimpleObject *obj = new (allocator) SimpleObject(allocator);
    expectedAllocationSize += sizeof(SimpleObject) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    delete obj;
    expectedAllocationSize -= sizeof(SimpleObject) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST_F(AllocatorTest, test_object_with_stl) {
    std::shared_ptr<VecSimAllocator> allocator(VecSimAllocator::newVecsimAllocator());
    size_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    ObjectWithSTL *obj = new (allocator) ObjectWithSTL(allocator);
    expectedAllocationSize += sizeof(ObjectWithSTL) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    obj->test_vec.push_back(1);
    expectedAllocationSize += sizeof(int) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    delete obj;
}

TEST_F(AllocatorTest, test_nested_object) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    NestedObject *obj = new (allocator) NestedObject(allocator);
    expectedAllocationSize += sizeof(NestedObject) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    obj->stl_object.test_vec.push_back(1);
    expectedAllocationSize += sizeof(int) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    delete obj;
}

template <typename index_type_t>
class IndexAllocatorTest : public ::testing::Test {};

// DataTypeSet, TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

TYPED_TEST_SUITE(IndexAllocatorTest, DataTypeSet);

TYPED_TEST(IndexAllocatorTest, test_bf_index_block_size_1) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    // Create only the minimal struct.
    size_t dim = 128;
    size_t blockSize = 1;
    BFParams params = {.type = TypeParam::get_index_type(),
                       .dim = dim,
                       .metric = VecSimMetric_IP,
                       .initialCapacity = 0,
                       .blockSize = blockSize};

    TEST_DATA_T vec[128] = {};
    BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T> *bfIndex =
        new (allocator) BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T>(&params, allocator);
    expectedAllocationSize +=
        sizeof(BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T>) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    size_t memory = VecSimIndex_StatsInfo(bfIndex).memory;
    ASSERT_EQ(allocator->getAllocationSize(), memory);

    // @param expected_size - The expected number of elements in the index.
    // @param expected_data_container_blocks - The expected number of blocks in the data containers.
    // @param expected_map_containers_capacity - The expected capacity of the map containers in
    // number of elements.
    auto verify_containers_size = [&](size_t expected_size, size_t expected_data_container_blocks,
                                      size_t expected_map_containers_size) {
        ASSERT_EQ(bfIndex->indexSize(), expected_size);
        ASSERT_EQ(bfIndex->vectorBlocks.size(), expected_data_container_blocks);
        ASSERT_EQ(bfIndex->getStoredVectorsCount(), expected_size);

        ASSERT_EQ(bfIndex->indexCapacity(), expected_map_containers_size);
        ASSERT_EQ(bfIndex->idToLabelMapping.capacity(), expected_map_containers_size);
        ASSERT_EQ(bfIndex->idToLabelMapping.size(), expected_map_containers_size);
        ASSERT_GE(bfIndex->labelToIdLookup.bucket_count(), expected_map_containers_size);
    };
    // =========== Add label 1 ===========
    size_t buckets_num_before = bfIndex->labelToIdLookup.bucket_count();
    auto &vectors_blocks = bfIndex->vectorBlocks;
    size_t vectors_blocks_capacity = vectors_blocks.capacity();

    int addCommandAllocationDelta = VecSimIndex_AddVector(bfIndex, vec, 1);
    int64_t expectedAllocationDelta =
        sizeof(labelType) + vecsimAllocationOverhead; // resize idToLabelMapping
    expectedAllocationDelta +=
        (vectors_blocks.capacity() - vectors_blocks_capacity) * sizeof(VectorBlock *) +
        vecsimAllocationOverhead; // New vectors blocks pointers
    expectedAllocationDelta +=
        blockSize * sizeof(TEST_DATA_T) * dim + vecsimAllocationOverhead; // block vectors buffer
    expectedAllocationDelta +=
        sizeof(VectorBlock) + vecsimAllocationOverhead; // Keep the allocated vector block
    expectedAllocationDelta += hashTableNodeSize;       // New node in the label lookup
    // Account for the allocation of a new buckets in the labels_lookup hash table.
    expectedAllocationDelta +=
        (bfIndex->labelToIdLookup.bucket_count() - buckets_num_before) * sizeof(size_t);
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    {
        SCOPED_TRACE("Verifying allocation delta for adding first vector");
        verify_containers_size(1, 1, 1);
        ASSERT_EQ(allocator->getAllocationSize(),
                  expectedAllocationSize + addCommandAllocationDelta);
        ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
        ASSERT_LE(expectedAllocationDelta, addCommandAllocationDelta);
        memory = VecSimIndex_StatsInfo(bfIndex).memory;
        ASSERT_EQ(allocator->getAllocationSize(), memory);
    }

    // =========== labels = [1], vector blocks = 1, maps capacity = 1. Add label 2 + 3 ===========

    // Prepare for next assertion test
    expectedAllocationSize = memory;
    expectedAllocationDelta = 0;

    vectors_blocks_capacity = vectors_blocks.capacity();
    buckets_num_before = bfIndex->labelToIdLookup.bucket_count();

    addCommandAllocationDelta = VecSimIndex_AddVector(bfIndex, vec, 2);
    addCommandAllocationDelta += VecSimIndex_AddVector(bfIndex, vec, 3);
    expectedAllocationDelta += (vectors_blocks.capacity() - vectors_blocks_capacity) *
                               sizeof(VectorBlock *); // New vector blocks pointers
    expectedAllocationDelta += 2 * sizeof(labelType); // resize idToLabelMapping
    expectedAllocationDelta += 2 * (blockSize * sizeof(TEST_DATA_T) * dim +
                                    vecsimAllocationOverhead); // Two block vectors buffer
    expectedAllocationDelta +=
        2 * (sizeof(VectorBlock) + vecsimAllocationOverhead); // Keep the allocated vector blocks
    expectedAllocationDelta += 2 * hashTableNodeSize;         // New nodes in the label lookup
    expectedAllocationDelta +=
        (bfIndex->labelToIdLookup.bucket_count() - buckets_num_before) * sizeof(size_t);
    {
        SCOPED_TRACE("Index size = Verifying allocation delta for adding two more vectors");
        verify_containers_size(3, 3, 3);
        ASSERT_EQ(allocator->getAllocationSize(),
                  expectedAllocationSize + addCommandAllocationDelta);
        ASSERT_EQ(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
        ASSERT_EQ(expectedAllocationDelta, addCommandAllocationDelta);
        memory = VecSimIndex_StatsInfo(bfIndex).memory;
        ASSERT_EQ(allocator->getAllocationSize(), memory);
    }

    // =========== labels = [1, 2, 3], vector blocks = 3, maps capacity = 3. Delete label 1
    // ===========

    // Prepare for next assertion test
    expectedAllocationSize = memory;
    expectedAllocationDelta = 0;

    vectors_blocks_capacity = vectors_blocks.capacity();
    buckets_num_before = bfIndex->labelToIdLookup.bucket_count();
    {
        SCOPED_TRACE("Verifying allocation delta for deleting a vector from index size 3");
        int deleteCommandAllocationDelta = VecSimIndex_DeleteVector(bfIndex, 1);
        verify_containers_size(2, 2, 3);
        // Removing blocks doesn't change vectors_blocks.capacity(), but the block buffer is freed.
        ASSERT_EQ(vectors_blocks.capacity(), vectors_blocks_capacity);
        expectedAllocationDelta -=
            (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block
        expectedAllocationDelta -=
            blockSize * sizeof(TEST_DATA_T) * dim +
            vecsimAllocationOverhead;                 // Free the vector buffer in the vector block
        expectedAllocationDelta -= hashTableNodeSize; // Remove node from the label lookup
        // idToLabelMapping and label:id should not change since count > capacity - 2 * blockSize
        ASSERT_EQ(bfIndex->labelToIdLookup.bucket_count(), buckets_num_before);

        ASSERT_EQ(allocator->getAllocationSize(),
                  expectedAllocationSize + deleteCommandAllocationDelta);
        ASSERT_EQ(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
        ASSERT_EQ(expectedAllocationDelta, deleteCommandAllocationDelta);

        memory = VecSimIndex_StatsInfo(bfIndex).memory;
        ASSERT_EQ(allocator->getAllocationSize(), memory);
    }

    // =========== labels = [2, 3], vector blocks = 2, maps capacity = 3. Add label 4 ===========

    // Prepare for next assertion test
    expectedAllocationSize = memory;
    expectedAllocationDelta = 0;

    vectors_blocks_capacity = vectors_blocks.capacity();
    buckets_num_before = bfIndex->labelToIdLookup.bucket_count();
    size_t idToLabel_size_before = bfIndex->idToLabelMapping.size();

    addCommandAllocationDelta = VecSimIndex_AddVector(bfIndex, vec, 4);
    expectedAllocationDelta += (vectors_blocks.capacity() - vectors_blocks_capacity) *
                               sizeof(VectorBlock *); // New vector block pointers
    expectedAllocationDelta +=
        sizeof(VectorBlock) + vecsimAllocationOverhead; // Keep the allocated vector blocks

    expectedAllocationDelta +=
        blockSize * sizeof(TEST_DATA_T) * dim + vecsimAllocationOverhead; // block vectors buffer
    expectedAllocationDelta += hashTableNodeSize; // New node in the label lookup
    {
        SCOPED_TRACE(
            "Verifying allocation delta for adding a vector to index size 2 with capacity 3");
        verify_containers_size(3, 3, 3);
        ASSERT_EQ(allocator->getAllocationSize(),
                  expectedAllocationSize + addCommandAllocationDelta);
        ASSERT_EQ(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
        ASSERT_EQ(expectedAllocationDelta, addCommandAllocationDelta);
        memory = VecSimIndex_StatsInfo(bfIndex).memory;
        ASSERT_EQ(allocator->getAllocationSize(), memory);

        // idToLabelMapping and label:id should not change since if we one free block
        ASSERT_EQ(bfIndex->labelToIdLookup.bucket_count(), buckets_num_before);
        ASSERT_EQ(bfIndex->idToLabelMapping.size(), idToLabel_size_before);
    }

    // =========== labels = [2, 3, 4], vector blocks = 3, maps capacity = 3. Delete label 2 + 3
    // ===========

    // Prepare for next assertion test
    expectedAllocationSize = memory;
    expectedAllocationDelta = 0;

    vectors_blocks_capacity = vectors_blocks.capacity();
    buckets_num_before = bfIndex->labelToIdLookup.bucket_count();
    {
        SCOPED_TRACE("Verifying allocation delta for deleting two vectors from index size 3");
        int deleteCommandAllocationDelta = VecSimIndex_DeleteVector(bfIndex, 2);
        deleteCommandAllocationDelta += VecSimIndex_DeleteVector(bfIndex, 3);
        verify_containers_size(1, 1, 2);
        // Removing blocks doesn't change vectors_blocks.capacity(), but the block buffer is freed.
        ASSERT_EQ(vectors_blocks.capacity(), vectors_blocks_capacity);
        expectedAllocationDelta -=
            2 * (blockSize * sizeof(TEST_DATA_T) * dim +
                 vecsimAllocationOverhead); // Free the vector buffer in the vector block
        expectedAllocationDelta -=
            2 * (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block

        expectedAllocationDelta -= 2 * hashTableNodeSize; // Remove nodes from the label lookup
        // idToLabelMapping and label:id should shrink by block since count >= capacity - 2 *
        // blockSize
        expectedAllocationDelta -= sizeof(labelType); // remove one idToLabelMapping
        expectedAllocationDelta -=
            (buckets_num_before - bfIndex->labelToIdLookup.bucket_count()) * sizeof(size_t);
        ASSERT_EQ(allocator->getAllocationSize(),
                  expectedAllocationSize + deleteCommandAllocationDelta);
        ASSERT_EQ(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
        ASSERT_EQ(expectedAllocationDelta, deleteCommandAllocationDelta);

        memory = VecSimIndex_StatsInfo(bfIndex).memory;
        ASSERT_EQ(allocator->getAllocationSize(), memory);
    }

    // =========== labels = [4], vector blocks = 1, maps capacity = 2. Delete last label ===========

    // Prepare for next assertion test
    expectedAllocationSize = memory;
    expectedAllocationDelta = 0;

    vectors_blocks_capacity = vectors_blocks.capacity();
    buckets_num_before = bfIndex->labelToIdLookup.bucket_count();
    {
        SCOPED_TRACE("Verifying allocation delta for emptying the index");
        int deleteCommandAllocationDelta = VecSimIndex_DeleteVector(bfIndex, 4);

        // We decrease meta data containers size by one block
        verify_containers_size(0, 0, 1);
        // Removing blocks doesn't change vectors_blocks.capacity(), but the block buffer is freed.
        ASSERT_EQ(vectors_blocks.capacity(), vectors_blocks_capacity);
        expectedAllocationDelta -=
            blockSize * sizeof(TEST_DATA_T) * dim +
            vecsimAllocationOverhead; // Free the vector buffer in the vector block
        expectedAllocationDelta -=
            (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block
        expectedAllocationDelta -= hashTableNodeSize;         // Remove nodes from the label lookup
        // idToLabelMapping and label:id should shrink by block since count >= capacity - 2 *
        // blockSize
        expectedAllocationDelta -=
            sizeof(labelType); // remove one idToLabelMapping and free the container
        // resizing labelToIdLookup
        size_t buckets_after = bfIndex->labelToIdLookup.bucket_count();
        expectedAllocationDelta -= (buckets_num_before - buckets_after) * sizeof(size_t);
        ASSERT_EQ(allocator->getAllocationSize(),
                  expectedAllocationSize + deleteCommandAllocationDelta);
        ASSERT_LE(abs(expectedAllocationDelta), abs(deleteCommandAllocationDelta));
        ASSERT_GE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());

        memory = VecSimIndex_StatsInfo(bfIndex).memory;
        ASSERT_EQ(allocator->getAllocationSize(), memory);
    }

    VecSimIndex_Free(bfIndex);
}

TYPED_TEST(IndexAllocatorTest, test_hnsw) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    size_t d = 128;

    // Build with default args
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 0};

    TEST_DATA_T vec[128] = {};
    HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *hnswIndex =
        new (allocator) HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T>(&params, allocator);
    expectedAllocationSize +=
        sizeof(HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T>) + vecsimAllocationOverhead;
    ASSERT_GE(allocator->getAllocationSize(), expectedAllocationSize);
    size_t memory = VecSimIndex_StatsInfo(hnswIndex).memory;
    ASSERT_EQ(allocator->getAllocationSize(), memory);
    expectedAllocationSize = memory;

    int addCommandAllocationDelta = VecSimIndex_AddVector(hnswIndex, vec, 1);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    memory = VecSimIndex_StatsInfo(hnswIndex).memory;
    ASSERT_EQ(allocator->getAllocationSize(), memory);
    expectedAllocationSize = memory;

    addCommandAllocationDelta = VecSimIndex_AddVector(hnswIndex, vec, 2);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    memory = VecSimIndex_StatsInfo(hnswIndex).memory;
    ASSERT_EQ(allocator->getAllocationSize(), memory);

    expectedAllocationSize = memory;

    int deleteCommandAllocationDelta = VecSimIndex_DeleteVector(hnswIndex, 2);
    ASSERT_EQ(expectedAllocationSize + deleteCommandAllocationDelta,
              allocator->getAllocationSize());
    memory = VecSimIndex_StatsInfo(hnswIndex).memory;
    ASSERT_EQ(allocator->getAllocationSize(), memory);
    expectedAllocationSize = memory;

    deleteCommandAllocationDelta = VecSimIndex_DeleteVector(hnswIndex, 1);
    ASSERT_EQ(expectedAllocationSize + deleteCommandAllocationDelta,
              allocator->getAllocationSize());
    memory = VecSimIndex_StatsInfo(hnswIndex).memory;
    ASSERT_EQ(allocator->getAllocationSize(), memory);
    VecSimIndex_Free(hnswIndex);
}

TYPED_TEST(IndexAllocatorTest, testIncomingEdgesSet) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t d = 2;

    // Build index, use small M to simplify the scenario.
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 10,
                         .M = 2};
    auto *hnswIndex =
        new (allocator) HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T>(&params, allocator);

    // Add a "dummy" vector - labels_lookup hash table will allocate initial size of buckets here.
    GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, 0, 0.0);

    // Add another vector and validate it's exact memory allocation delta.
    TEST_DATA_T vec1[] = {1.0, 0.0};
    int allocation_delta = VecSimIndex_AddVector(hnswIndex, vec1, 1);
    size_t vec_max_level = hnswIndex->element_levels_[1];

    // Expect the creation of an empty incoming edges set in every level (+ the allocator header
    // overhead), and a single node in the labels' lookup hash table.
    size_t expected_allocation_delta =
        (vec_max_level + 1) * (sizeof(vecsim_stl::vector<idType>) + vecsimAllocationOverhead);
    expected_allocation_delta += hashTableNodeSize;

    // Account for allocating link lists for levels higher than 0, if exists.
    if (vec_max_level > 0) {
        expected_allocation_delta +=
            hnswIndex->size_links_per_element_ * vec_max_level + vecsimAllocationOverhead;
    }
    ASSERT_EQ(allocation_delta, expected_allocation_delta);

    // Add three more vectors, all should have a connections to vec1.
    TEST_DATA_T vec2[] = {2.0f, 0.0f};
    VecSimIndex_AddVector(hnswIndex, vec2, 2);
    TEST_DATA_T vec3[] = {1.0f, 1.0f};
    VecSimIndex_AddVector(hnswIndex, vec3, 3);
    TEST_DATA_T vec4[] = {1.0f, -1.0f};
    VecSimIndex_AddVector(hnswIndex, vec4, 4);

    // Layer 0 should look like this (all edges bidirectional):
    //    3                    3
    //    |                    |
    // 0--1--2      =>   0--5--1--2
    //    |              |----^|
    //    4                    4

    // Next, insertion of vec5 should make 0->1 unidirectional, thus adding 0 to 1's incoming edges
    // set.
    TEST_DATA_T vec5[] = {0.5f, 0.0f};
    size_t buckets_num_before = hnswIndex->label_lookup_.bucket_count();
    allocation_delta = VecSimIndex_AddVector(hnswIndex, vec5, 5);
    vec_max_level = hnswIndex->element_levels_[5];

    /* Compute the expected allocation delta:
     * 1. empty incoming edges set in every level (+ allocator's header).
     * 2. A node in the labels_lookup has table (+ allocator's header). If rehashing occurred, we
     * account also for the diff in the buckets size (each bucket has sizeof(size_t) overhead).
     * 3. Account for allocating link lists for levels higher than 0, if exists.
     * 4. Finally, expect an allocation of the data buffer in the incoming edges vector of vec1 due
     * to the insertion, and the fact that vec1 will re-select its neighbours.
     */
    expected_allocation_delta =
        (vec_max_level + 1) * (sizeof(vecsim_stl::vector<idType>) + vecsimAllocationOverhead) +
        hashTableNodeSize;
    size_t buckets_diff = hnswIndex->label_lookup_.bucket_count() - buckets_num_before;
    expected_allocation_delta += buckets_diff * sizeof(size_t);
    if (vec_max_level > 0) {
        expected_allocation_delta +=
            hnswIndex->size_links_per_element_ * vec_max_level + vecsimAllocationOverhead;
    }

    // Expect that the first element is pushed to the incoming edges vector of element 1 in level 0.
    // Then, we account for the capacity of the buffer that is allocated for the vector data.
    expected_allocation_delta += hnswIndex->getIncomingEdgesPtr(1, 0)->capacity() * sizeof(idType) +
                                 vecsimAllocationOverhead;
    ASSERT_EQ(allocation_delta, expected_allocation_delta);

    VecSimIndex_Free(hnswIndex);
}

TYPED_TEST(IndexAllocatorTest, test_hnsw_reclaim_memory) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t d = 128;

    VecSimType type = TypeParam::get_index_type();

    // Build HNSW index with default args and initial capacity of zero.
    HNSWParams params = {.type = type, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0};
    auto *hnswIndex =
        new (allocator) HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T>(&params, allocator);

    ASSERT_EQ(hnswIndex->indexCapacity(), 0);
    size_t initial_memory_size = allocator->getAllocationSize();
    // labels_lookup and element_levels containers are not allocated at all in some platforms,
    // when initial capacity is zero, while in other platforms labels_lookup is allocated with a
    // single bucket. This, we get the following range in which we expect the initial memory to be
    // in.
    ASSERT_GE(initial_memory_size, HNSWFactory::EstimateInitialSize(&params));
    ASSERT_LE(initial_memory_size, HNSWFactory::EstimateInitialSize(&params) + sizeof(size_t) +
                                       2 * vecsimAllocationOverhead);

    // Add vectors up to the size of a whole block, and calculate the total memory delta.
    size_t block_size = hnswIndex->debugInfo().hnswInfo.blockSize;
    // Allocations caused by adding a new vector:
    auto compute_vec_mem = [&](idType id) {
        // Compute the expected memory allocation due to the  last vector insertion.
        size_t vec_max_level = hnswIndex->element_levels_[id];
        // Incoming edges
        size_t new_vec_mem_delta =
            (vec_max_level + 1) * (sizeof(vecsim_stl::vector<idType>) + vecsimAllocationOverhead);
        if (vec_max_level > 0) {
            new_vec_mem_delta +=
                hnswIndex->size_links_per_element_ * vec_max_level + vecsimAllocationOverhead;
        }
        // new node in the labels_lookup hash table
        new_vec_mem_delta += hashTableNodeSize;

        return new_vec_mem_delta;
    };
    // Add the first vector to store the first block allocation delta.
    size_t initial_one_block_mem_delta = GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, 0, 0);
    size_t one_block_mem_delta = initial_one_block_mem_delta;
    size_t initial_buckets_count = hnswIndex->label_lookup_.bucket_count();
    for (size_t i = 1; i < block_size; i++) {
        one_block_mem_delta += GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, i, i);
    }

    // Remove the memory allocated for the first vector, since it is added as a "dummy" vector.
    initial_one_block_mem_delta -= compute_vec_mem(0);
    size_t one_block_buckets = hnswIndex->label_lookup_.bucket_count();
    // @param expected_size - The expected number of elements in the index.
    // @param expected_capacity - The expected capacity in elements.
    auto verify_containers_size = [&](size_t expected_size, size_t expected_capacity) {
        SCOPED_TRACE("Verifying containers size for size " + std::to_string(expected_size));
        ASSERT_EQ(hnswIndex->indexSize(), expected_size);
        ASSERT_EQ(hnswIndex->indexCapacity(), expected_capacity);
        ASSERT_EQ(hnswIndex->indexCapacity(), hnswIndex->max_elements_);
        ASSERT_EQ(hnswIndex->element_levels_.size(), expected_capacity);
        ASSERT_EQ(hnswIndex->element_levels_.size(), hnswIndex->element_levels_.capacity());

        ASSERT_GE(hnswIndex->label_lookup_.bucket_count(), expected_capacity);
        // Also validate that there are no unidirectional connections (these add memory to the
        // incoming edges sets).
        ASSERT_EQ(hnswIndex->checkIntegrity().unidirectional_connections, 0);
    };

    // Validate that a single block exists.
    verify_containers_size(block_size, block_size);
    ASSERT_EQ(allocator->getAllocationSize(), initial_memory_size + one_block_mem_delta);

    // Add another vector, expect resizing of the index to contain two blocks.
    size_t prev_bucket_count = hnswIndex->label_lookup_.bucket_count();
    size_t mem_delta = GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, block_size, block_size);
    verify_containers_size(block_size + 1, 2 * block_size);

    // Allocations caused by adding a block:
    // element_levels, data_level0_memory_, linkLists_, visitedNodesHandlerPool
    size_t containers_mem =
        (sizeof(size_t) + hnswIndex->size_data_per_element_ + sizeof(void *) + sizeof(tag_t)) *
        block_size;
    int hash_table_mem_delta =
        (hnswIndex->label_lookup_.bucket_count() - prev_bucket_count) * sizeof(size_t);
    size_t add_one_block_mem_delta = containers_mem + hash_table_mem_delta;
    size_t new_vec_mem_delta = compute_vec_mem(block_size);

    ASSERT_EQ(add_one_block_mem_delta + new_vec_mem_delta, mem_delta);

    // Remove the last vector, since size + 2 * block_size > capacity, expect the containers size to
    // NOT change. Only the vector's memory is freed.
    int delete_vec_mem_delta = VecSimIndex_DeleteVector(hnswIndex, block_size);
    verify_containers_size(block_size, 2 * block_size);

    ASSERT_EQ(static_cast<size_t>(-delete_vec_mem_delta), new_vec_mem_delta);

    // Remove the rest of the vectors, meta data containers size should decrease by one block.
    prev_bucket_count = hnswIndex->label_lookup_.bucket_count();
    size_t expected_delete_mem_delta = 0;
    int delete_block_mem_delta = 0;
    for (int i = block_size - 1; i >= 0; i--) {
        expected_delete_mem_delta += compute_vec_mem(i);
        delete_block_mem_delta += VecSimIndex_DeleteVector(hnswIndex, i);
    }
    verify_containers_size(0, block_size);

    hash_table_mem_delta =
        (prev_bucket_count - hnswIndex->label_lookup_.bucket_count()) * sizeof(size_t);
    expected_delete_mem_delta += hash_table_mem_delta + containers_mem;

    ASSERT_EQ(static_cast<size_t>(-delete_block_mem_delta), expected_delete_mem_delta);

    // Adding a vector should not cause new containers memory, only the vector's memory.
    mem_delta = GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, 0);
    verify_containers_size(1, block_size);

    ASSERT_EQ(mem_delta, new_vec_mem_delta);
    ASSERT_EQ(initial_memory_size + initial_one_block_mem_delta + mem_delta,
              allocator->getAllocationSize());

    // Delete the new (and only vec)
    // Since the index is empty, and the capacity equals block_size, the containers should shrink to
    // 0.
    prev_bucket_count = hnswIndex->label_lookup_.bucket_count();
    delete_vec_mem_delta = VecSimIndex_DeleteVector(hnswIndex, 0);
    verify_containers_size(0, 0);
    hash_table_mem_delta =
        (initial_buckets_count - hnswIndex->label_lookup_.bucket_count()) * sizeof(size_t);
    ASSERT_LE(hnswIndex->label_lookup_.bucket_count(), initial_buckets_count);
    ASSERT_LE(containers_mem, initial_one_block_mem_delta);
    ASSERT_GE(initial_memory_size +
                  (initial_one_block_mem_delta - containers_mem - hash_table_mem_delta),
              allocator->getAllocationSize());
    hash_table_mem_delta =
        (prev_bucket_count - hnswIndex->label_lookup_.bucket_count()) * sizeof(size_t);
    expected_delete_mem_delta = hash_table_mem_delta + containers_mem + mem_delta;
    ASSERT_LE(expected_delete_mem_delta, static_cast<size_t>(-delete_vec_mem_delta));

    VecSimIndex_Free(hnswIndex);
}
