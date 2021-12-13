#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/spaces/space_interface.h"

class AllocatorTest : public ::testing::Test {
protected:
    AllocatorTest() {}

    ~AllocatorTest() override {}

    void SetUp() override {}

    void TearDown() override {}

    static uint64_t vecsimAllocationOverhead;
};

uint64_t AllocatorTest::vecsimAllocationOverhead = sizeof(size_t);

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
                       .size = dim,
                       .metric = VecSimMetric_IP,
                       .initialCapacity = 0,
                       .blockSize = 1};

    float vec[128] = {};
    BruteForceIndex *bfIndex = new (allocator) BruteForceIndex(&params, allocator);
    expectedAllocationSize +=
        sizeof(BruteForceIndex) + sizeof(InnerProductSpace) + 2 * vecsimAllocationOverhead;
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    VecSimIndexInfo info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    bfIndex->addVector(vec, 1);
    uint64_t allocationDelta = 0;
    allocationDelta += 2 * ((sizeof(VectorBlockMember *) +
                             vecsimAllocationOverhead)); // resize idToVectorBlockMemberMapping to 2
    allocationDelta += sizeof(VectorBlock) + vecsimAllocationOverhead; // New vector block
    allocationDelta += sizeof(VectorBlockMember) + vecsimAllocationOverhead;
    allocationDelta += sizeof(VectorBlockMember *) +
                       vecsimAllocationOverhead; // Pointer for the new vector block member
    allocationDelta +=
        sizeof(float) * dim + vecsimAllocationOverhead; // keep the vector in the vector block
    allocationDelta +=
        sizeof(VectorBlock *) + vecsimAllocationOverhead; // Keep the allocated vector block
    allocationDelta +=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // keep the mapping
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_TRUE(expectedAllocationSize + allocationDelta <= allocator->getAllocationSize() &&
                allocator->getAllocationSize() <= expectedAllocationSize + allocationDelta * 2);

    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.memory;
    allocationDelta = 0;

    bfIndex->addVector(vec, 2);
    allocationDelta += sizeof(VectorBlock) + vecsimAllocationOverhead; // New vector block
    allocationDelta += sizeof(VectorBlockMember) + vecsimAllocationOverhead;
    allocationDelta += sizeof(VectorBlockMember *) +
                       vecsimAllocationOverhead; // Pointer for the new vector block member
    allocationDelta +=
        sizeof(float) * dim + vecsimAllocationOverhead; // keep the vector in the vector block
    allocationDelta +=
        sizeof(VectorBlock *) + vecsimAllocationOverhead; // Keep the allocated vector block
    allocationDelta +=
        sizeof(std::pair<labelType, idType>) + vecsimAllocationOverhead; // keep the mapping
    // Assert that the additional allocated delta did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_TRUE(expectedAllocationSize + allocationDelta <= allocator->getAllocationSize() &&
                allocator->getAllocationSize() <= expectedAllocationSize + allocationDelta * 2);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.memory;
    allocationDelta = 0;

    bfIndex->deleteVector(2);
    allocationDelta -= (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block
    allocationDelta -= (sizeof(VectorBlockMember) + vecsimAllocationOverhead);
    allocationDelta -= (sizeof(VectorBlockMember *) +
                        vecsimAllocationOverhead); // Pointer for the new vector block member
    allocationDelta -=
        (sizeof(float) * dim + vecsimAllocationOverhead); // Free the vector in the vector block

    // Assert that the reclaiming of memory did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_TRUE(expectedAllocationSize >= allocator->getAllocationSize() &&
                allocator->getAllocationSize() >= expectedAllocationSize + allocationDelta);

    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    // Prepare for next assertion test
    expectedAllocationSize = info.memory;
    allocationDelta = 0;

    bfIndex->deleteVector(1);
    allocationDelta -= (sizeof(VectorBlock) + vecsimAllocationOverhead); // Free the vector block
    allocationDelta -= (sizeof(VectorBlockMember) + vecsimAllocationOverhead);
    allocationDelta -= (sizeof(VectorBlockMember *) +
                        vecsimAllocationOverhead); //  Pointer for the new vector block member
    allocationDelta -=
        (sizeof(float) * dim + vecsimAllocationOverhead); // Free the vector in the vector block
    // Assert that the reclaiming of memory did occur, and it is limited, as some STL
    // collection allocate additional structures for their internal implementation.
    ASSERT_TRUE(expectedAllocationSize >= allocator->getAllocationSize() &&
                allocator->getAllocationSize() >= expectedAllocationSize + allocationDelta);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    VecSimIndex_Free(bfIndex);
}

TEST_F(AllocatorTest, test_hnsw) {
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    uint64_t expectedAllocationSize = sizeof(VecSimAllocator);
    ASSERT_EQ(allocator->getAllocationSize(), expectedAllocationSize);
    size_t d = 128;

    // Build with default args
    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .size = d,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 0};

    float vec[128] = {};
    HNSWIndex *hnswIndex = new (allocator) HNSWIndex(&params, allocator);
    expectedAllocationSize +=
        sizeof(HNSWIndex) + sizeof(InnerProductSpace) + 2 * vecsimAllocationOverhead;
    ASSERT_GE(allocator->getAllocationSize(), expectedAllocationSize);
    VecSimIndexInfo info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    expectedAllocationSize = info.memory;

    hnswIndex->addVector(vec, 1);
    ASSERT_GE(allocator->getAllocationSize(), expectedAllocationSize);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    expectedAllocationSize = info.memory;

    hnswIndex->addVector(vec, 2);
    ASSERT_GE(allocator->getAllocationSize(), expectedAllocationSize);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    VecSimIndex_Free(hnswIndex);

    // TODO: commented out since hnsw does not recalim memory
    // current_memory = info.memory;

    // hnswIndex->deleteVector(2);
    // ASSERT_GE(current_memory, allocator->getAllocationSize());
    // info = hnswIndex->info();
    // ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    // current_memory = info.memory;

    // hnswIndex->deleteVector(1);
    // ASSERT_GE(current_memory, allocator->getAllocationSize());
    // info = hnswIndex->info();
    // ASSERT_EQ(allocator->getAllocationSize(), info.memory);
}
