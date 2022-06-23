#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
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

TEST_F(AllocatorTest, test_hnsw_reclaim_memory) {
	std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
	size_t d = 128;

	// Build with default args
	HNSWParams params = {
			.type = VecSimType_FLOAT32, .dim = d, .metric = VecSimMetric_L2, .initialCapacity = 0};
	auto *hnswIndex = new (allocator) HNSWIndex(&params, allocator);
	size_t block_size = 1024; // currently, this is the default and is not configurable.

	// get memory (to compare later)

	// Add vectors up to the size of a whole block.
	for (size_t i = 0; i < block_size; i++) {
		float vec[d];
		for (size_t j = 0; j < d; j++) {
			vec[j] = (float)rand();
		}
		VecSimIndex_AddVector(hnswIndex, vec, i);
	}
	//todo:
	// validate one block exists
	// add some vectors
	// validate two blocks exit
	// remove the vectors in arbitrary order
	// validate one block exist + memory returns to what it was
	// remove the rest of the vectors
	// validate memory back to what it was.
}
