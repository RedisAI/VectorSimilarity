#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "test_utils.h"
#include "VecSim/algorithms/hnsw/serialization.h"

class AllocatorTest : public ::testing::Test {
protected:
    AllocatorTest() {}

    ~AllocatorTest() override {}

    void SetUp() override {}

    void TearDown() override {}

    static uint64_t vecsimAllocationOverhead;

    static uint64_t hashTableNodeSize;

    static uint64_t setNodeSize;
};

uint64_t AllocatorTest::vecsimAllocationOverhead = sizeof(size_t);

uint64_t AllocatorTest::hashTableNodeSize = getLabelsLookupNodeSize();

uint64_t AllocatorTest::setNodeSize = getIncomingEdgesSetNodeSize();

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
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
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
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
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
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    NestedObject *obj = new (allocator) NestedObject(allocator);
    expectedAllocationSize += sizeof(NestedObject) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    obj->stl_object.test_vec.push_back(1);
    expectedAllocationSize += sizeof(int) + vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    delete obj;
}

TEST_F(AllocatorTest, test_bf_index_block_size_1) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    // Create only the minimal struct.
    size_t dim = 128;
    BFParams params = {.type = VecSimType_FLOAT32,
                       .dim = dim,
                       .metric = VecSimMetric_IP,
                       .initialCapacity = 0,
                       .blockSize = 1};

    float vec[128] = {};
    BruteForceIndex *bfIndex = new (allocator) BruteForceIndex(&params, allocator);
    expectedAllocationSize +=
        sizeof(BruteForceIndex) + sizeof(InnerProductSpace) + 2 * vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    VecSimIndexInfo info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.bfInfo.memory);

    int addCommandAllocationDelta = VecSimIndex_AddVector(bfIndex, vec, 1);
    int64_t expectedAllocationDelta = 0;
    expectedAllocationDelta +=
        2 * ((sizeof(VectorBlockMember *) +
              vecsimAllocationOverhead)); // resize idToVectorBlockMemberMapping to 2
    expectedAllocationDelta += sizeof(VectorBlock) + vecsimAllocationOverhead; // New vector block
    expectedAllocationDelta += sizeof(VectorBlockMember) + vecsimAllocationOverhead;
    expectedAllocationDelta += sizeof(VectorBlockMember *) +
                               vecsimAllocationOverhead; // Pointer for the new vector block member
    expectedAllocationDelta +=
        sizeof(float) * dim + vecsimAllocationOverhead; // keep the vector in the vector block
    expectedAllocationDelta +=
        sizeof(VectorBlock *) + vecsimAllocationOverhead; // Keep the allocated vector block
    expectedAllocationDelta +=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // keep the mapping
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, addCommandAllocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.bfInfo.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.bfInfo.memory;
    expectedAllocationDelta = 0;

    addCommandAllocationDelta = VecSimIndex_AddVector(bfIndex, vec, 2);
    expectedAllocationDelta += sizeof(VectorBlock) + vecsimAllocationOverhead; // New vector block
    expectedAllocationDelta += sizeof(VectorBlockMember) + vecsimAllocationOverhead;
    expectedAllocationDelta += sizeof(VectorBlockMember *) +
                               vecsimAllocationOverhead; // Pointer for the new vector block member
    expectedAllocationDelta +=
        sizeof(float) * dim + vecsimAllocationOverhead; // keep the vector in the vector block
    expectedAllocationDelta +=
        sizeof(VectorBlock *) + vecsimAllocationOverhead; // Keep the allocated vector block
    expectedAllocationDelta +=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // keep the mapping
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, addCommandAllocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.bfInfo.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.bfInfo.memory;
    expectedAllocationDelta = 0;

    int deleteCommandAllocationDelta = VecSimIndex_DeleteVector(bfIndex, 2);
    expectedAllocationDelta -=
        (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block
    expectedAllocationDelta -= (sizeof(VectorBlockMember) + vecsimAllocationOverhead);
    expectedAllocationDelta -=
        (sizeof(VectorBlockMember *) +
         vecsimAllocationOverhead); // Pointer for the new vector block member
    expectedAllocationDelta -=
        (sizeof(float) * dim + vecsimAllocationOverhead); // Free the vector in the vector block

    // Assert that the reclaiming of memory did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(),
              expectedAllocationSize + deleteCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, deleteCommandAllocationDelta);

    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.bfInfo.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.bfInfo.memory;
    expectedAllocationDelta = 0;

    deleteCommandAllocationDelta = VecSimIndex_DeleteVector(bfIndex, 1);
    expectedAllocationDelta -=
        (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block
    expectedAllocationDelta -= (sizeof(VectorBlockMember) + vecsimAllocationOverhead);
    expectedAllocationDelta -=
        (sizeof(VectorBlockMember *) +
         vecsimAllocationOverhead); //  Pointer for the new vector block member
    expectedAllocationDelta -=
        (sizeof(float) * dim + vecsimAllocationOverhead); // Free the vector in the vector block
    // Assert that the reclaiming of memory did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_EQ(allocator->getAllocationSize(),
              expectedAllocationSize + deleteCommandAllocationDelta);
    ASSERT_LE(expectedAllocationSize + expectedAllocationDelta, allocator->getAllocationSize());
    ASSERT_LE(expectedAllocationDelta, deleteCommandAllocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.bfInfo.memory);
    VecSimIndex_Free(bfIndex);
}

namespace hnswlib {

TEST_F(AllocatorTest, test_hnsw) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    size_t d = 128;

    // Build with default args
    HNSWParams params = {
        .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0};

    float vec[128] = {};
    HNSWIndex *hnswIndex = new (allocator) HNSWIndex(&params, allocator);
    expectedAllocationSize +=
        sizeof(HNSWIndex) + sizeof(InnerProductSpace) + 2 * vecsimAllocationOverhead;
    ASSERT_GE(allocator->getAllocationSize(), expectedAllocationSize);
    VecSimIndexInfo info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.hnswInfo.memory);
    expectedAllocationSize = info.hnswInfo.memory;

    int addCommandAllocationDelta = VecSimIndex_AddVector(hnswIndex, vec, 1);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.hnswInfo.memory);
    expectedAllocationSize = info.hnswInfo.memory;

    addCommandAllocationDelta = VecSimIndex_AddVector(hnswIndex, vec, 2);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize + addCommandAllocationDelta);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.hnswInfo.memory);

    expectedAllocationSize = info.hnswInfo.memory;

    int deleteCommandAllocationDelta = VecSimIndex_DeleteVector(hnswIndex, 2);
    ASSERT_EQ(expectedAllocationSize + deleteCommandAllocationDelta,
              allocator->getAllocationSize());
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.hnswInfo.memory);
    expectedAllocationSize = info.hnswInfo.memory;

    deleteCommandAllocationDelta = VecSimIndex_DeleteVector(hnswIndex, 1);
    ASSERT_EQ(expectedAllocationSize + deleteCommandAllocationDelta,
              allocator->getAllocationSize());
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.hnswInfo.memory);
    VecSimIndex_Free(hnswIndex);
}

TEST_F(AllocatorTest, testIncomingEdgesSet) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t d = 2;

    // Build index, use small M to simplify the scenario.
    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .dim = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 10,
                         .M = 2};
    auto *hnswIndex = new (allocator) HNSWIndex(&params, allocator);

    // Add a "dummy" vector - labels_lookup hash table will allocate initial size of buckets here.
    float vec0[] = {0.0f, 0.0f};
    VecSimIndex_AddVector(hnswIndex, vec0, 0);

    // Add another vector and validate it's exact memory allocation delta.
    float vec1[] = {1.0f, 0.0f};
    int allocation_delta = VecSimIndex_AddVector(hnswIndex, vec1, 1);
    size_t vec_max_level = hnswIndex->getHNSWIndex()->element_levels_[1];

    // Expect the creation of an empty incoming edges set in every level (+ the allocator header
    // overhead), and a single node in the labels' lookup hash table.
    size_t expected_allocation_delta =
        (vec_max_level + 1) *
        (sizeof(vecsim_stl::set<hnswlib::tableint>) + AllocatorTest::vecsimAllocationOverhead);
    expected_allocation_delta += AllocatorTest::hashTableNodeSize;

    // Account for allocating link lists for levels higher than 0, if exists.
    if (vec_max_level > 0) {
        expected_allocation_delta +=
            hnswIndex->getHNSWIndex()->size_links_per_element_ * vec_max_level + 1 +
            AllocatorTest::vecsimAllocationOverhead;
    }
    ASSERT_EQ(allocation_delta, expected_allocation_delta);

    // Add three more vectors, all should have a connections to vec1.
    float vec2[] = {2.0f, 0.0f};
    VecSimIndex_AddVector(hnswIndex, vec2, 2);
    float vec3[] = {1.0f, 1.0f};
    VecSimIndex_AddVector(hnswIndex, vec3, 3);
    float vec4[] = {1.0f, -1.0f};
    VecSimIndex_AddVector(hnswIndex, vec4, 4);

    // Layer 0 should look like this (all edges bidirectional):
    //    3                    3
    //    |                    |
    // 0--1--2      =>   0--5--1--2
    //    |              |----^|
    //    4                    4

    // Next, insertion of vec5 should make 0->1 unidirectional, thus adding 0 to 1's incoming edges
    // set.
    float vec5[] = {0.5f, 0.0f};
    size_t buckets_num_before = hnswIndex->getHNSWIndex()->label_lookup_.bucket_count();
    allocation_delta = VecSimIndex_AddVector(hnswIndex, vec5, 5);
    vec_max_level = hnswIndex->getHNSWIndex()->element_levels_[5];

    /* Compute the expected allocation delta:
     * 1. empty incoming edges set in every level (+ allocator's header).
     * 2. A node in the labels_lookup has table (+ allocator's header). If rehashing occurred, we
     * account also for the diff in the buckets size (each bucket has sizeof(size_t) overhead).
     * 3. Account for allocating link lists for levels higher than 0, if exists.
     * 4. Finally, expect an allocation of a node in the incoming edges set of vec1 due to the
     * insertion, and the fact that vec1 will re-select its neighbours.
     */
    expected_allocation_delta = (vec_max_level + 1) * (sizeof(vecsim_stl::set<hnswlib::tableint>) +
                                                       AllocatorTest::vecsimAllocationOverhead) +
                                AllocatorTest::hashTableNodeSize;
    size_t buckets_diff =
        hnswIndex->getHNSWIndex()->label_lookup_.bucket_count() - buckets_num_before;
    expected_allocation_delta += buckets_diff * sizeof(size_t);
    if (vec_max_level > 0) {
        expected_allocation_delta +=
            hnswIndex->getHNSWIndex()->size_links_per_element_ * vec_max_level + 1 +
            AllocatorTest::vecsimAllocationOverhead;
    }
    expected_allocation_delta += AllocatorTest::setNodeSize;
    ASSERT_EQ(allocation_delta, expected_allocation_delta);

    VecSimIndex_Free(hnswIndex);
}

TEST_F(AllocatorTest, test_hnsw_reclaim_memory) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    size_t d = 128;

    // Build HNSW index with default args and initial capacity of zero.
    HNSWParams params = {
        .type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0};
    auto *hnswIndex = new (allocator) HNSWIndex(&params, allocator);

    ASSERT_EQ(hnswIndex->getHNSWIndex()->getIndexCapacity(), 0);
    size_t initial_memory_size = allocator->getAllocationSize();
    // labels_lookup and element_levels containers are not allocated at all in some platforms,
    // when initial capacity is zero, while in other platforms labels_lookup is allocated with a
    // single bucket. This, we get the following range in which we expect the initial memory to be
    // in.
    ASSERT_LE(initial_memory_size, HNSWIndex::estimateInitialSize(&params) + sizeof(size_t));
    ASSERT_GE(initial_memory_size,
              HNSWIndex::estimateInitialSize(&params) - 2 * vecsimAllocationOverhead);

    // Add vectors up to the size of a whole block, and calculate the total memory delta.
    size_t block_size = hnswIndex->info().hnswInfo.blockSize;
    size_t accumulated_mem_delta = 0;
    float vec[d];
    for (size_t i = 0; i < block_size; i++) {
        for (size_t j = 0; j < d; j++) {
            vec[j] = (float)i;
        }
        accumulated_mem_delta += VecSimIndex_AddVector(hnswIndex, vec, i);
    }
    // Validate that a single block exists.
    ASSERT_EQ(hnswIndex->indexSize(), block_size);
    ASSERT_EQ(hnswIndex->getHNSWIndex()->getIndexCapacity(), block_size);
    ASSERT_EQ(allocator->getAllocationSize(), initial_memory_size + accumulated_mem_delta);
    // Also validate that there are no unidirectional connections (these add memory to the incoming
    // edges sets).
    auto serializer = HNSWIndexSerializer(hnswIndex->getHNSWIndex());
    ASSERT_EQ(serializer.checkIntegrity().unidirectional_connections, 0);

    // Add another vector, expect resizing of the index to contain two blocks.
    for (size_t j = 0; j < d; j++) {
        vec[j] = (float)block_size;
    }
    VecSimIndex_AddVector(hnswIndex, vec, block_size);
    ASSERT_EQ(hnswIndex->indexSize(), block_size + 1);
    ASSERT_EQ(hnswIndex->getHNSWIndex()->getIndexCapacity(), 2 * block_size);
    ASSERT_EQ(serializer.checkIntegrity().unidirectional_connections, 0);

    // Remove the last vector, expect resizing back to a single block, and return to the previous
    // memory consumption.
    VecSimIndex_DeleteVector(hnswIndex, block_size);
    ASSERT_EQ(hnswIndex->indexSize(), block_size);
    ASSERT_EQ(hnswIndex->getHNSWIndex()->getIndexCapacity(), block_size);
    ASSERT_EQ(serializer.checkIntegrity().unidirectional_connections, 0);
    ASSERT_EQ(allocator->getAllocationSize(), initial_memory_size + accumulated_mem_delta);

    // Remove the rest of the vectors, and validate that the memory returns to its initial state.
    for (size_t i = 0; i < block_size; i++) {
        VecSimIndex_DeleteVector(hnswIndex, i);
    }

    ASSERT_EQ(hnswIndex->indexSize(), 0);
    ASSERT_EQ(hnswIndex->getHNSWIndex()->getIndexCapacity(), 0);
    // All data structures' memory returns to as it was, with the exceptional of the labels_lookup
    // (STL unordered_map with hash table implementation), that leaves some empty buckets.
    size_t hash_table_memory =
        hnswIndex->getHNSWIndex()->label_lookup_.bucket_count() * sizeof(size_t);
    // current memory should be in the following range, depends on whether the label_lookup
    // hash table was allocated with a single bucket at start or not. If so, the diff from the
    // initial memory size should be at most the current hash_table_memory - size of a single
    // bucket. On the other hand, the diff may also include the label_lookup and element_levels
    // headers.
    ASSERT_GE(allocator->getAllocationSize(),
              initial_memory_size + hash_table_memory - sizeof(size_t));
    ASSERT_LE(allocator->getAllocationSize(),
              initial_memory_size + hash_table_memory + 2 * vecsimAllocationOverhead);

    VecSimIndex_Free(hnswIndex);
}

} // namespace hnswlib
