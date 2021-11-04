#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnswlib_c.h"

class AllocatorTest : public ::testing::Test {
protected:
    AllocatorTest() {}

    ~AllocatorTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

struct SimpleObject : public VecsimBaseObject {
public:
    SimpleObject(VecSimAllocator *allocator) : VecsimBaseObject(allocator) {}
    SimpleObject(std::shared_ptr<VecSimAllocator> allocator) : VecsimBaseObject(allocator) {}
    int x;
};

struct ObjectWithSTL : public VecsimBaseObject {
    std::vector<int, VecsimSTLAllocator<int>> test_vec;

public:
    ObjectWithSTL(VecSimAllocator *allocator) : VecsimBaseObject(allocator), test_vec(allocator){};
    ObjectWithSTL(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), test_vec(allocator){};
};

struct NestedObject : public VecsimBaseObject {
    ObjectWithSTL stl_object;
    SimpleObject simpleObject;

public:
    NestedObject(VecSimAllocator *allocator)
        : VecsimBaseObject(allocator), stl_object(allocator), simpleObject(allocator){};
    NestedObject(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), stl_object(allocator), simpleObject(allocator){};
};

TEST_F(AllocatorTest, test_simple_object) {
    std::shared_ptr<VecSimAllocator> allocator = std::make_shared<VecSimAllocator>();
    SimpleObject *obj = new (allocator) SimpleObject(allocator);
    ASSERT_EQ(allocator->getAllocationSize(), sizeof(SimpleObject) + sizeof(VecSimAllocator));
    delete obj;
    ASSERT_EQ(allocator->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST_F(AllocatorTest, test_object_with_stl) {
    std::shared_ptr<VecSimAllocator> allocator = std::make_shared<VecSimAllocator>();
    ObjectWithSTL *obj = new (allocator) ObjectWithSTL(allocator);
    ASSERT_EQ(allocator->getAllocationSize(), sizeof(ObjectWithSTL) + sizeof(VecSimAllocator));
    obj->test_vec.push_back(1);
    ASSERT_EQ(allocator->getAllocationSize(),
              sizeof(ObjectWithSTL) + sizeof(VecSimAllocator) + sizeof(int));
}

TEST_F(AllocatorTest, test_nested_object) {
    std::shared_ptr<VecSimAllocator> allocator = std::make_shared<VecSimAllocator>();
    NestedObject *obj = new (allocator) NestedObject(allocator);
    ASSERT_EQ(allocator->getAllocationSize(), sizeof(NestedObject) + sizeof(VecSimAllocator));
    obj->stl_object.test_vec.push_back(1);
    ASSERT_EQ(allocator->getAllocationSize(),
              sizeof(NestedObject) + sizeof(VecSimAllocator) + sizeof(int));
}

TEST_F(AllocatorTest, test_bf_index_block_size_1) {
    std::shared_ptr<VecSimAllocator> allocator = std::make_shared<VecSimAllocator>();
    // Create only the minimal struct.
    size_t dim = 128;
    VecSimParams params = {.bfParams = {.initialCapacity = 0, .blockSize = 1},
                           .type = VecSimType_FLOAT32,
                           .size = dim,
                           .metric = VecSimMetric_IP,
                           .algo = VecSimAlgo_BF};

    float vec[128] = {};
    BruteForceIndex *bfIndex = new (allocator) BruteForceIndex(&params, allocator);
    ASSERT_GE(allocator->getAllocationSize(), sizeof(VecSimAllocator) + sizeof(BruteForceIndex));
    VecSimIndexInfo info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    bfIndex->addVector(vec, 1);
    size_t allocations = 0;
    allocations += sizeof(VecSimAllocator);         // Create allocator
    allocations += sizeof(BruteForceIndex);         // Create index struct
    allocations += 2 * sizeof(VectorBlockMember *); // resize idToVectorBlockMemberMapping to 2
    allocations += sizeof(VectorBlock);             // New vector block
    allocations += sizeof(VectorBlockMember);
    allocations += sizeof(VectorBlockMember *);          // Pointer for the new vector block member
    allocations += sizeof(float) * dim;                  // keep the vector in the vector block
    allocations += sizeof(VectorBlock *);                // Keep the allocated vector block
    allocations += sizeof(std::pair<labelType, idType>); // keep the mapping
    ASSERT_GE(allocator->getAllocationSize(), allocations);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    bfIndex->addVector(vec, 2);
    allocations += 2 * sizeof(VectorBlockMember *); // resize idToVectorBlockMemberMapping to 4
    allocations += sizeof(VectorBlock);             // New vector block
    allocations += sizeof(VectorBlockMember);
    allocations += sizeof(VectorBlockMember *);          // Pointer for the new vector block member
    allocations += sizeof(float) * dim;                  // keep the vector in the vector block
    allocations += sizeof(VectorBlock *);                // Keep the allocated vector block
    allocations += sizeof(std::pair<labelType, idType>); // keep the mapping
    ASSERT_GE(allocator->getAllocationSize(), allocations);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    bfIndex->deleteVector(2);
    allocations -= sizeof(VectorBlock); // New vector block
    allocations -= sizeof(VectorBlockMember);
    allocations -= sizeof(VectorBlockMember *); // Pointer for the new vector block member
    allocations -= sizeof(float) * dim;         // keep the vector in the vector block
    ASSERT_GE(allocator->getAllocationSize(), allocations);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    bfIndex->deleteVector(1);
    allocations -= sizeof(VectorBlock); // New vector block
    allocations -= sizeof(VectorBlockMember);
    allocations -= sizeof(VectorBlockMember *); // Pointer for the new vector block member
    allocations -= sizeof(float) * dim;         // keep the vector in the vector block
    ASSERT_GE(allocator->getAllocationSize(), allocations);
    info = bfIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    delete bfIndex;
}

TEST_F(AllocatorTest, test_hnsw) {
    std::shared_ptr<VecSimAllocator> allocator = std::make_shared<VecSimAllocator>();
    size_t d = 128;

    // Build with default args
    VecSimParams params = {
        .hnswParams =
            {
                .initialCapacity = 0,
            },
        .type = VecSimType_FLOAT32,
        .size = d,
        .metric = VecSimMetric_L2,
        .algo = VecSimAlgo_HNSWLIB,
    };

    float vec[128] = {};
    HNSWIndex *hnswIndex = new (allocator) HNSWIndex(&params, allocator);
    ASSERT_GE(allocator->getAllocationSize(), sizeof(VecSimAllocator) + sizeof(HNSWIndex));
    VecSimIndexInfo info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    size_t current_memory = info.memory;

    hnswIndex->addVector(vec, 1);
    ASSERT_GE(allocator->getAllocationSize(), current_memory);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
    current_memory = info.memory;

    hnswIndex->addVector(vec, 2);
    ASSERT_GE(allocator->getAllocationSize(), current_memory);
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    hnswIndex->deleteVector(2);
    ASSERT_GE(current_memory, allocator->getAllocationSize());
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);

    hnswIndex->deleteVector(1);
    ASSERT_GE(current_memory, allocator->getAllocationSize());
    info = hnswIndex->info();
    ASSERT_EQ(allocator->getAllocationSize(), info.memory);
}
