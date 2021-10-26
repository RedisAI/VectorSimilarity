#include "gtest/gtest.h"
#include "VecSim/vecsim.h"
#include "VecSim/memory/vecsim_malloc.h"

class AllocatorTest : public ::testing::Test {
protected:
    AllocatorTest() {}

    ~AllocatorTest() override {}

    void SetUp() override {}

    void TearDown() override {}
};

struct SimpleObject : public VecsimBaseObject {
public:
    int x;

    virtual size_t size() { return sizeof(SimpleObject); }
};

struct ObjectWithSTL : public VecsimBaseObject {
    std::vector<int, VecsimAllocator<int>> test_vec;

public:
    ObjectWithSTL(VecsimAllocator<VecsimBaseObject> allocator)
        : VecsimBaseObject(allocator), test_vec(allocator){};

    virtual size_t size() { return *(this->getAllocator()).allocated; };
};

TEST_F(AllocatorTest, test_simple_object) {
    VecsimAllocator<VecsimBaseObject> allocator;
    SimpleObject *obj = new (allocator) SimpleObject;
    ASSERT_EQ(*allocator.allocated.get(), sizeof(SimpleObject) + sizeof(allocator.allocated));
}

TEST_F(AllocatorTest, test_object_with_stl) {
    VecsimAllocator<VecsimBaseObject> allocator;
    ObjectWithSTL *obj = new (allocator) ObjectWithSTL(allocator);
    ASSERT_EQ(*allocator.allocated.get(), sizeof(ObjectWithSTL) + sizeof(allocator.allocated));
    ASSERT_EQ(*allocator.allocated.get(), obj->size());
    obj->test_vec.push_back(1);
    ASSERT_EQ(*allocator.allocated.get(),
              sizeof(ObjectWithSTL) + sizeof(allocator.allocated) + sizeof(int));
    ASSERT_EQ(*allocator.allocated.get(), obj->size());
}