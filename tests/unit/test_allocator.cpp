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
#include "VecSim/index_factories/hnsw_factory.h"

const size_t vecsimAllocationOverhead = VecSimAllocator::getAllocationOverheadSize();

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
    // Create only the minimal struct.
    size_t dim = 128;
    BFParams params = {.type = TypeParam::get_index_type(),
                       .dim = dim,
                       .metric = VecSimMetric_IP,
                       .initialCapacity = 0,
                       .blockSize = 1};
    auto *bfIndex = dynamic_cast<BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(
        BruteForceFactory::NewIndex(&params));
    bfIndex->alignment = 0; // Disable alignment for testing purposes.
    auto allocator = bfIndex->getAllocator();
    TEST_DATA_T vec[128] = {};
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    expectedAllocationSize +=
        sizeof(BruteForceIndex_Single<TEST_DATA_T, TEST_DIST_T>) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    VecSimIndexInfo info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);

    int before = allocator->getAllocationSize();
    VecSimIndex_AddVector(bfIndex, vec, 1);
    int addCommandAllocationDelta = allocator->getAllocationSize() - before;
    int64_t expectedAllocationDelta = 0;
    expectedAllocationDelta +=
        sizeof(labelType) + vecsimAllocationOverhead; // resize idToLabelMapping
    expectedAllocationDelta += sizeof(DataBlock) + vecsimAllocationOverhead; // New vector block
    expectedAllocationDelta +=
        sizeof(TEST_DATA_T) * dim + vecsimAllocationOverhead; // keep the vector in the vector block
    expectedAllocationDelta +=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // keep the mapping
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, addCommandAllocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.commonInfo.memory;
    expectedAllocationDelta = 0;

    before = allocator->getAllocationSize();
    VecSimIndex_AddVector(bfIndex, vec, 2);
    addCommandAllocationDelta = allocator->getAllocationSize() - before;
    expectedAllocationDelta += sizeof(DataBlock) + vecsimAllocationOverhead; // New vector block
    expectedAllocationDelta += sizeof(labelType); // resize idToLabelMapping
    expectedAllocationDelta +=
        sizeof(TEST_DATA_T) * dim + vecsimAllocationOverhead; // keep the vector in the vector block
    expectedAllocationDelta +=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // keep the mapping
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, addCommandAllocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.commonInfo.memory;
    expectedAllocationDelta = 0;

    before = allocator->getAllocationSize();
    VecSimIndex_DeleteVector(bfIndex, 2);
    int deleteCommandAllocationDelta = allocator->getAllocationSize() - before;
    expectedAllocationDelta -=
        sizeof(TEST_DATA_T) * dim + vecsimAllocationOverhead; // Free the vector in the vector block
    expectedAllocationDelta -= sizeof(labelType);             // resize idToLabelMapping
    expectedAllocationDelta -=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // remove one label:id pair

    // Assert that the reclaiming of memory did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(),
              expectedAllocationSize + deleteCommandAllocationDelta);
    ASSERT_GE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_GE(expectedAllocationDelta, deleteCommandAllocationDelta);

    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.commonInfo.memory;
    expectedAllocationDelta = 0;

    before = allocator->getAllocationSize();
    VecSimIndex_DeleteVector(bfIndex, 1);
    deleteCommandAllocationDelta = allocator->getAllocationSize() - before;
    expectedAllocationDelta -=
        (sizeof(DataBlock) + vecsimAllocationOverhead); // Free the vector block
    expectedAllocationDelta -=
        sizeof(DataBlock *) + vecsimAllocationOverhead; // remove from vectorBlocks vector
    expectedAllocationDelta -=
        sizeof(labelType) + vecsimAllocationOverhead; // resize idToLabelMapping
    expectedAllocationDelta -= (sizeof(TEST_DATA_T) * dim +
                                vecsimAllocationOverhead); // Free the vector in the vector block
    expectedAllocationDelta -=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // remove one label:id pair

    // Assert that the reclaiming of memory did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(),
              expectedAllocationSize + deleteCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, deleteCommandAllocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);
    VecSimIndex_Free(bfIndex);
}

TYPED_TEST(IndexAllocatorTest, test_hnsw) {
    size_t d = 128;

    // Build with default args
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 0};

    TEST_DATA_T vec[128] = {};
    auto *hnswIndex =
        dynamic_cast<HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(HNSWFactory::NewIndex(&params));
    auto allocator = hnswIndex->getAllocator();
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);

    expectedAllocationSize +=
        sizeof(HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T>) + vecsimAllocationOverhead;
    ASSERT_GE(allocator->getAllocationSize(), expectedAllocationSize);
    VecSimIndexInfo info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);
    expectedAllocationSize = info.commonInfo.memory;

    int before = allocator->getAllocationSize();
    VecSimIndex_AddVector(hnswIndex, vec, 1);
    int addCommandAllocationDelta = allocator->getAllocationSize() - before;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);
    expectedAllocationSize = info.commonInfo.memory;

    before = allocator->getAllocationSize();
    VecSimIndex_AddVector(hnswIndex, vec, 2);
    addCommandAllocationDelta = allocator->getAllocationSize() - before;

    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);

    expectedAllocationSize = info.commonInfo.memory;

    before = allocator->getAllocationSize();
    VecSimIndex_DeleteVector(hnswIndex, 2);
    int deleteCommandAllocationDelta = allocator->getAllocationSize() - before;

    ASSERT_EQ(expectedAllocationSize + deleteCommandAllocationDelta,
              allocator->getAllocationSize());
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);
    expectedAllocationSize = info.commonInfo.memory;

    before = allocator->getAllocationSize();
    VecSimIndex_DeleteVector(hnswIndex, 1);
    deleteCommandAllocationDelta = allocator->getAllocationSize() - before;

    ASSERT_EQ(expectedAllocationSize + deleteCommandAllocationDelta,
              allocator->getAllocationSize());
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.commonInfo.memory);
    VecSimIndex_Free(hnswIndex);
}

TYPED_TEST(IndexAllocatorTest, testIncomingEdgesSet) {

    size_t d = 2;

    // Build index, use small M to simplify the scenario.
    HNSWParams params = {.type = TypeParam::get_index_type(),
                         .dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 10,
                         .M = 2};
    auto *hnswIndex =
        dynamic_cast<HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(HNSWFactory::NewIndex(&params));
    auto allocator = hnswIndex->getAllocator();

    // Add a "dummy" vector - labels_lookup hash table will allocate initial size of buckets here.
    GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, 0, 0.0);

    // Add another vector and validate it's exact memory allocation delta.
    TEST_DATA_T vec1[] = {1.0, 0.0};
    int before = allocator->getAllocationSize();
    VecSimIndex_AddVector(hnswIndex, vec1, 1);
    int allocation_delta = allocator->getAllocationSize() - before;
    size_t vec_max_level = hnswIndex->getGraphDataByInternalId(1)->toplevel;

    // Expect the creation of an empty incoming edges set in every level (+ the allocator header
    // overhead), and a single node in the labels' lookup hash table.
    size_t expected_allocation_delta =
        (vec_max_level + 1) * (sizeof(vecsim_stl::vector<idType>) + vecsimAllocationOverhead);
    expected_allocation_delta += hashTableNodeSize;

    // Account for allocating link lists for levels higher than 0, if exists.
    if (vec_max_level > 0) {
        expected_allocation_delta +=
            hnswIndex->levelDataSize * vec_max_level + vecsimAllocationOverhead;
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
    size_t buckets_num_before = hnswIndex->labelLookup.bucket_count();
    before = allocator->getAllocationSize();
    VecSimIndex_AddVector(hnswIndex, vec5, 5);
    allocation_delta = allocator->getAllocationSize() - before;
    vec_max_level = hnswIndex->getGraphDataByInternalId(5)->toplevel;

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
    size_t buckets_diff = hnswIndex->labelLookup.bucket_count() - buckets_num_before;
    expected_allocation_delta += buckets_diff * sizeof(size_t);
    if (vec_max_level > 0) {
        expected_allocation_delta +=
            hnswIndex->levelDataSize * vec_max_level + vecsimAllocationOverhead;
    }

    // Expect that the first element is pushed to the incoming edges vector of element 1 in level 0.
    // Then, we account for the capacity of the buffer that is allocated for the vector data.
    expected_allocation_delta +=
        hnswIndex->getLevelData(1, 0).incomingEdges->capacity() * sizeof(idType) +
        vecsimAllocationOverhead;
    ASSERT_EQ(allocation_delta, expected_allocation_delta);

    VecSimIndex_Free(hnswIndex);
}

TYPED_TEST(IndexAllocatorTest, test_hnsw_reclaim_memory) {
    size_t d = 128;

    VecSimType type = TypeParam::get_index_type();

    // Build HNSW index with default args and initial capacity of zero.
    HNSWParams params = {.type = type, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0};
    auto *hnswIndex =
        dynamic_cast<HNSWIndex_Single<TEST_DATA_T, TEST_DIST_T> *>(HNSWFactory::NewIndex(&params));
    auto allocator = hnswIndex->getAllocator();
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
    size_t block_size = hnswIndex->info().commonInfo.basicInfo.blockSize;

    size_t accumulated_mem_delta = allocator->getAllocationSize();
    for (size_t i = 0; i < block_size; i++) {
        GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, i, i);
    }
    // Get the memory delta after adding the block.
    accumulated_mem_delta = allocator->getAllocationSize() - accumulated_mem_delta;

    // Validate that a single block exists.
    ASSERT_EQ(hnswIndex->indexSize(), block_size);
    ASSERT_EQ(hnswIndex->indexCapacity(), block_size);
    ASSERT_EQ(allocator->getAllocationSize(), initial_memory_size + accumulated_mem_delta);
    // Also validate that there are no unidirectional connections (these add memory to the incoming
    // edges sets).
    ASSERT_EQ(hnswIndex->checkIntegrity().unidirectional_connections, 0);

    // Add another vector, expect resizing of the index to contain two blocks.
    size_t prev_bucket_count = hnswIndex->labelLookup.bucket_count();
    size_t mem_delta = allocator->getAllocationSize();
    GenerateAndAddVector<TEST_DATA_T>(hnswIndex, d, block_size, block_size);
    mem_delta = allocator->getAllocationSize() - mem_delta;

    ASSERT_EQ(hnswIndex->indexSize(), block_size + 1);
    ASSERT_EQ(hnswIndex->indexCapacity(), 2 * block_size);
    ASSERT_EQ(hnswIndex->checkIntegrity().unidirectional_connections, 0);

    // Compute the expected memory allocation due to the last vector insertion.
    size_t vec_max_level = hnswIndex->getGraphDataByInternalId(block_size)->toplevel;
    size_t expected_mem_delta =
        (vec_max_level + 1) * (sizeof(vecsim_stl::vector<idType>) + vecsimAllocationOverhead) +
        hashTableNodeSize;
    if (vec_max_level > 0) {
        expected_mem_delta += hnswIndex->levelDataSize * vec_max_level + vecsimAllocationOverhead;
    }
    // Also account for all the memory allocation caused by the resizing that this vector triggered
    // except for the bucket count of the labels_lookup hash table that is calculated separately.
    size_t size_total_data_per_element = hnswIndex->elementGraphDataSize + hnswIndex->dataSize;
    expected_mem_delta +=
        (sizeof(tag_t) + sizeof(labelType) + sizeof(elementFlags) + size_total_data_per_element) *
        block_size;
    expected_mem_delta +=
        (hnswIndex->labelLookup.bucket_count() - prev_bucket_count) * sizeof(size_t);
    // New blocks allocated - 1 aligned block for vectors and 1 unaligned block for graph data.
    expected_mem_delta += 2 * (sizeof(DataBlock) + vecsimAllocationOverhead) + hnswIndex->alignment;
    expected_mem_delta +=
        (hnswIndex->vectorBlocks.capacity() - hnswIndex->vectorBlocks.size()) * sizeof(DataBlock);
    expected_mem_delta +=
        (hnswIndex->graphDataBlocks.capacity() - hnswIndex->graphDataBlocks.size()) *
        sizeof(DataBlock);

    ASSERT_EQ(expected_mem_delta, mem_delta);

    // Remove the last vector, expect resizing back to a single block, and return to the previous
    // memory consumption.
    VecSimIndex_DeleteVector(hnswIndex, block_size);
    ASSERT_EQ(hnswIndex->indexSize(), block_size);
    ASSERT_EQ(hnswIndex->indexCapacity(), block_size);
    ASSERT_EQ(hnswIndex->checkIntegrity().unidirectional_connections, 0);
    size_t expected_allocation_size = initial_memory_size + accumulated_mem_delta;
    expected_allocation_size +=
        (hnswIndex->vectorBlocks.capacity() - hnswIndex->vectorBlocks.size()) * sizeof(DataBlock);
    expected_allocation_size +=
        (hnswIndex->graphDataBlocks.capacity() - hnswIndex->graphDataBlocks.size()) *
        sizeof(DataBlock);
    ASSERT_EQ(allocator->getAllocationSize(), expected_allocation_size);

    // Remove the rest of the vectors, and validate that the memory returns to its initial state.
    for (size_t i = 0; i < block_size; i++) {
        VecSimIndex_DeleteVector(hnswIndex, i);
    }

    ASSERT_EQ(hnswIndex->indexSize(), 0);
    ASSERT_EQ(hnswIndex->indexCapacity(), 0);
    // All data structures' memory returns to as it was, with the exceptional of the labels_lookup
    // (STL unordered_map with hash table implementation), that leaves some empty buckets.
    size_t hash_table_memory = hnswIndex->labelLookup.bucket_count() * sizeof(size_t);
    // Data block vectors do not shrink on resize so extra memory is expected.
    size_t block_vectors_memory = sizeof(DataBlock) * (hnswIndex->graphDataBlocks.capacity() +
                                                       hnswIndex->vectorBlocks.capacity()) +
                                  2 * vecsimAllocationOverhead;
    // Current memory should be back as it was initially. The label_lookup hash table is an
    // exception, since in some platforms, empty buckets remain even when the capacity is set to
    // zero, while in others the entire capacity reduced to zero (including the header).
    ASSERT_LE(allocator->getAllocationSize(), HNSWFactory::EstimateInitialSize(&params) +
                                                  block_vectors_memory + hash_table_memory +
                                                  2 * vecsimAllocationOverhead);
    ASSERT_GE(allocator->getAllocationSize(),
              HNSWFactory::EstimateInitialSize(&params) + block_vectors_memory + hash_table_memory);
    VecSimIndex_Free(hnswIndex);
}
