#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/updatable_heap.h"
#include "VecSim/utils/vec_utils.h"
#include "test_utils.h"

#include <cstdlib>
#include <limits>
#include <cmath>
#include <random>

template <typename index_type_t>
class CommonIndexTest : public ::testing::Test {};

// DataTypeSet are defined in test_utils.h

TYPED_TEST_SUITE(CommonIndexTest, DataTypeSet);

TYPED_TEST(CommonIndexTest, ResolveQueryRuntimeParams) {
    size_t dim = 4;

    BFParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = 0, .blockSize = 5};
    VecSimIndex *index = test_utils::CreateNewIndex(params, TypeParam::get_index_type());

    VecSimQueryParams qparams, zero;
    bzero(&zero, sizeof(VecSimQueryParams));

    auto *rparams = array_new<VecSimRawParam>(2);

    // Empty raw params array, nothing should change in query params.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSim_OK);
    ASSERT_EQ(memcmp(&qparams, &zero, sizeof(VecSimQueryParams)), 0);

    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), nullptr, false),
              VecSimParamResolverErr_NullParam);

    /** Testing with common hybrid query params. **/
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "100",
                                           .valLen = strlen("100")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "batch_size".
    array_append(rparams, (VecSimRawParam){.name = "batch_size",
                                           .nameLen = strlen("batch_size"),
                                           .value = "200",
                                           .valLen = strlen("200")});
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_AlreadySet);

    rparams[1] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches_wrong",
                                  .valLen = strlen("batches_wrong")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_InvalidPolicy_NExits);

    rparams[1].value = "batches";
    rparams[1].valLen = strlen("batches");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Both params are "hybrid policy".
    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "ADhOC_bf",
                                  .valLen = strlen("ADhOC_bf")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_AlreadySet);

    // Sending HYBRID_POLICY=adhoc as the single parameter is valid.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, 1, &qparams, true), VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_ADHOC_BF);

    // Cannot set batch_size param with "hybrid_policy" which is "ADHOC_BF"
    rparams[1] = (VecSimRawParam){.name = "batch_size",
                                  .nameLen = strlen("batch_size"),
                                  .value = "100",
                                  .valLen = strlen("100")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_InvalidPolicy_AdHoc_With_BatchSize);

    rparams[0] = (VecSimRawParam){.name = "HYBRID_POLICY",
                                  .nameLen = strlen("HYBRID_POLICY"),
                                  .value = "batches",
                                  .valLen = strlen("batches")};
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSim_OK);
    ASSERT_EQ(qparams.searchMode, HYBRID_BATCHES);
    ASSERT_EQ(qparams.batchSize, 100);

    // Trying to set hybrid policy for non-hybrid query.
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, false),
              VecSimParamResolverErr_InvalidPolicy_NHybrid);
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams + 1, 1, &qparams, false),
              VecSimParamResolverErr_InvalidPolicy_NHybrid);

    // Check for invalid batch sizes params.
    rparams[1].value = "not_a_number";
    rparams[1].valLen = strlen("not_a_number");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "9223372036854775808"; // LLONG_MAX+1
    rparams[1].valLen = strlen("9223372036854775808");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "-5";
    rparams[1].valLen = strlen("-5");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "0";
    rparams[1].valLen = strlen("0");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    rparams[1].value = "10f";
    rparams[1].valLen = strlen("10f");
    ASSERT_EQ(VecSimIndex_ResolveParams(index, rparams, array_len(rparams), &qparams, true),
              VecSimParamResolverErr_BadValue);

    VecSimIndex_Free(index);
    array_free(rparams);
}

template <typename DataType>
class UtilsTests : public ::testing::Test {};

using DataTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UtilsTests, DataTypes);

TYPED_TEST(UtilsTests, Max_Updatable_Heap) {
    std::pair<TypeParam, size_t> p;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();

    vecsim_stl::updatable_max_heap<TypeParam, size_t> heap(allocator);

    // Initial state checks
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());
    ASSERT_NO_THROW(heap.top());

    // Insert some data in random order
    size_t riders[] = {46, 16, 99, 93};
    const size_t n_riders = sizeof(riders) / sizeof(riders[0]);
    enum Priority { FIRST = 0, SECOND = 1, THIRD = 2, FOURTH = 3 };
    const TypeParam priorities[] = {M_PI, M_E, M_SQRT2, -M_SQRT2 * M_E};

    heap.emplace(priorities[THIRD], riders[1]);
    heap.emplace(priorities[FIRST], riders[3]);
    heap.emplace(priorities[SECOND], riders[2]);
    heap.emplace(priorities[FOURTH], riders[0]);

    for (size_t i = 0; i < n_riders; ++i) {
        ASSERT_EQ(heap.size(), n_riders - i);
        p = {priorities[i], riders[n_riders - 1 - i]};
        ASSERT_TRUE(heap.top() == p);
        ASSERT_FALSE(heap.empty());
        heap.pop();
    }

    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());

    // Inserting data with the same priority
    heap.emplace(priorities[SECOND], 1);
    heap.emplace(priorities[FIRST], 55);
    heap.emplace(priorities[SECOND], 3);
    heap.emplace(priorities[SECOND], 2);

    ASSERT_EQ(heap.size(), 4);
    ASSERT_FALSE(heap.empty());
    p = {priorities[FIRST], 55};
    ASSERT_TRUE(heap.top() == p);

    heap.emplace(priorities[THIRD], 55); // Update priority

    ASSERT_EQ(heap.size(), 4); // Same size after update
    ASSERT_FALSE(heap.empty());

    // Make sure each pop deletes a single element, even if some have the same priority.
    size_t len = heap.size();
    for (size_t i = len; i > 0; i--) {
        ASSERT_EQ(heap.size(), i);
        ASSERT_FALSE(heap.empty());
        heap.pop();
    }
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());

    // Update a priority of an element that share its priority with many others.
    size_t last = 10;
    for (size_t i = 0; i <= last; i++) {
        heap.emplace(priorities[SECOND], i);
    }
    // Bound the existing elements with higher and lower priorities.
    heap.emplace(priorities[THIRD], 42);
    heap.emplace(priorities[FIRST], 46);
    size_t size = heap.size();

    // Update to the lowest priority
    heap.emplace(-priorities[THIRD], last);
    ASSERT_EQ(heap.size(), size);

    while (heap.size() > 1) {
        heap.pop();
    }
    ASSERT_EQ(heap.size(), 1);
    ASSERT_FALSE(heap.empty());
    p = {-priorities[THIRD], last};
    ASSERT_TRUE(heap.top() == p);
    heap.pop();
    ASSERT_EQ(heap.size(), 0);
    ASSERT_TRUE(heap.empty());
}

TYPED_TEST(UtilsTests, VecSim_Normalize_Vector) {
    const size_t dim = 1000;
    TypeParam v[dim];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> dis(0.0, (TypeParam)std::numeric_limits<int>::max());

    // generate random values - always generates the same values
    for (size_t i = 0; i < dim; ++i) {
        v[i] = dis(rng);
    }

    // Change some of the vector's values so that the sum of the squared vector's
    // values will overflow for floats but not for doubles.
    v[dim - 3] = exp(44);
    v[dim - 2] = exp(44);
    v[dim - 1] = exp(44);

    // Normalize the vector
    normalizeVector(v, dim);

    // Check that the normelized vector norm is 1
    TypeParam norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        norm += v[i] * v[i];
    }

    TypeParam one = 1.0;
    ASSERT_NEAR(one, norm, 0.0000001);
}
