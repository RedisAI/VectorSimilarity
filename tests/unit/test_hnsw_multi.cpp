#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "test_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/hnsw_multi.h"
#include <cmath>
#include <map>

template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

template <typename index_type_t>
class HNSWMultiTest : public ::testing::Test {
public:
    HNSWMultiTest() : params{} {
        params.type = index_type_t::get_index_type();
        params.multi = true;
    }
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    void GenerateVector(data_t *output, size_t dim, data_t value = 1.0) {
        for (size_t i = 0; i < dim; i++) {
            output[i] = (data_t)value;
        }
    }
    // Returns the memory addition after adding the vector to the index.
    int GenerateNAddVector(VecSimIndex *index, size_t dim, size_t id, data_t value = 1.0) {
        data_t v[dim];
        this->GenerateVector(v, dim, value); // i / 10 is in integer (take the "floor" value).
        return VecSimIndex_AddVector(index, v, id);
    }

    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }

    HNSWIndex_Multi<data_t, dist_t> *CastToHNSW_Multi(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex_Multi<data_t, dist_t> *>(index);
    }

    HNSWParams params;
};

#define TEST_DATA_T typename TypeParam::data_t
#define TEST_DIST_T typename TypeParam::dist_t
using DataTypeSet =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>, IndexType<VecSimType_FLOAT64, double>>;

TYPED_TEST_SUITE(HNSWMultiTest, DataTypeSet);

TYPED_TEST(HNSWMultiTest, vector_add_multiple_test) {
    size_t dim = 4;
    size_t rep = 5;

    this->params.dim = dim;
    this->params.metric = VecSimMetric_IP;
    this->params.initialCapacity = 200;

    VecSimIndex *index = CreateNewIndex(this->params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Adding multiple vectors under the same label
    for (size_t j = 0; j < rep; j++) {
        TEST_DATA_T a[dim];
        for (size_t i = 0; i < dim; i++) {
            a[i] = (TEST_DATA_T)i * j + i;
        }
        VecSimIndex_AddVector(index, a, 46);
    }

    ASSERT_EQ(VecSimIndex_IndexSize(index), rep);
    ASSERT_EQ(index->indexLabelCount(), 1);

    // Deleting the label. All the vectors should be deleted.
    VecSimIndex_DeleteVector(index, 46);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);
    ASSERT_EQ(index->indexLabelCount(), 0);

    VecSimIndex_Free(index);
}

// Test empty index edge cases.
TYPED_TEST(HNSWMultiTest, empty_index) {
    size_t dim = 4;
    size_t n = 20;
    size_t bs = 6;

    this->params.dim = dim;
    this->params.metric = VecSimMetric_L2;
    this->params.initialCapacity = n;
    this->params.blockSize = bs;

    VecSimIndex *index = CreateNewIndex(this->params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove from an empty index - should fail because label doesn't exist.
    VecSimIndex_DeleteVector(index, 0);

    // Add one vector multiple times.
    for (size_t i = 0; i < 3; i++) {
        this->GenerateNAddVector(index, dim, 1, 1.7);
    }

    // Try to remove it.
    VecSimIndex_DeleteVector(index, 1);

    // Size equals 0.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Try to remove it again.
    VecSimIndex_DeleteVector(index, 1);

    // Size should be stiil zero.
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWMultiTest, vector_search_test) {
    size_t dim = 4;
    size_t n = 1000;
    size_t n_labels = 100;
    size_t k = 11;

    this->params.dim = dim;
    this->params.metric = VecSimMetric_L2;
    this->params.initialCapacity = 200;

    VecSimIndex *index = CreateNewIndex(this->params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i % n_labels, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim, 50);

    auto verify_res = [&](size_t id, double score, size_t index) {
        size_t diff_id = (id > 50) ? (id - 50) : (50 - id);
        ASSERT_EQ(diff_id, (index + 1) / 2);
        ASSERT_EQ(score, (4 * ((index + 1) / 2) * ((index + 1) / 2)));
    };
    runTopKSearchTest(index, query, k, verify_res);
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWMultiTest, search_more_than_there_is) {
    size_t dim = 4;
    size_t n = 5;
    size_t perLabel = 3;
    size_t n_labels = ceil((float)n / perLabel);
    size_t k = 3;
    // This test add 5 vectors under 2 labels, and then query for 3 results.
    // We want to make sure we get only 2 results back (because the results should have unique
    // labels), although the index contains 5 vectors.

    this->params.dim = dim;
    this->params.metric = VecSimMetric_L2;

    VecSimIndex *index = CreateNewIndex(this->params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateNAddVector(index, dim, i / perLabel, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    TEST_DATA_T query[dim];
    this->GenerateVector(query, dim, 0);

    VecSimQueryResult_List res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_SCORE);
    ASSERT_EQ(VecSimQueryResult_Len(res), n_labels);
    auto it = VecSimQueryResult_List_GetIterator(res);
    for (size_t i = 0; i < n_labels; i++) {
        auto el = VecSimQueryResult_IteratorNext(it);
        ASSERT_EQ(VecSimQueryResult_GetScore(el), i * perLabel * i * perLabel * dim);
        labelType element_label = VecSimQueryResult_GetId(el);
        ASSERT_EQ(element_label, i);
        auto ids = this->CastToHNSW_Multi(index)->label_lookup_.at(element_label);
        for (size_t j = 0; j < ids.size(); j++) {
            // Verifying that each vector is labeled correctly.
            // ID is calculated according to insertion order.
            ASSERT_EQ(ids[j], i * perLabel + j);
        }
    }
    VecSimQueryResult_IteratorFree(it);
    VecSimQueryResult_Free(res);
    VecSimIndex_Free(index);
}
