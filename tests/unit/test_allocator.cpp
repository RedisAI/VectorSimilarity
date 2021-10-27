#include "gtest/gtest.h"
#include "VecSim/vecsim.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/memory/vecsim_base.h"

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
    VecSimAllocator allocator;
    std::shared_ptr<VecSimAllocator> allocator_ptr = std::make_shared<VecSimAllocator>(allocator);
    SimpleObject *obj = new (allocator_ptr) SimpleObject(allocator_ptr);
    ASSERT_EQ(allocator_ptr->getAllocationSize(), sizeof(SimpleObject) + sizeof(VecSimAllocator));
    delete obj;
    ASSERT_EQ(allocator_ptr->getAllocationSize(), sizeof(VecSimAllocator));
}

TEST_F(AllocatorTest, test_object_with_stl) {
    VecSimAllocator allocator;
    std::shared_ptr<VecSimAllocator> allocator_ptr = std::make_shared<VecSimAllocator>(allocator);
    ObjectWithSTL *obj = new (allocator_ptr) ObjectWithSTL(allocator_ptr);
    ASSERT_EQ(allocator.getAllocationSize(), sizeof(ObjectWithSTL) + sizeof(VecSimAllocator));
    obj->test_vec.push_back(1);
    ASSERT_EQ(allocator.getAllocationSize(),
              sizeof(ObjectWithSTL) + sizeof(VecSimAllocator) + sizeof(int));
}

TEST_F(AllocatorTest, test_nested_object) {
    VecSimAllocator allocator;
    std::shared_ptr<VecSimAllocator> allocator_ptr = std::make_shared<VecSimAllocator>(allocator);
    NestedObject *obj = new (allocator_ptr) NestedObject(allocator_ptr);
    ASSERT_EQ(allocator.getAllocationSize(), sizeof(NestedObject) + sizeof(VecSimAllocator));
    obj->stl_object.test_vec.push_back(1);
    ASSERT_EQ(allocator.getAllocationSize(),
              sizeof(NestedObject) + sizeof(VecSimAllocator) + sizeof(int));
}