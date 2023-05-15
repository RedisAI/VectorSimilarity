#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/raft_ivf/ivf_index_interface.h"
#include "test_utils.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/query_result_struct.h"
#include <climits>
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

template <typename index_type_t>
class RaftIvfTest : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *CreateNewIndex(RaftIVFFlatParams &params) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), false);
    }
    VecSimIndex *CreateNewIndex(RaftIVFPQParams &params) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), false);
    }
    RaftIvfIndexInterface *CastToInterface(VecSimIndex *index) {
        return dynamic_cast<RaftIvfIndexInterface *>(index);
    }
};

// TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

RaftIVFPQParams createDefaultPQParams(size_t dim)
{
    RaftIVFPQParams params = {.dim = dim,
                              .metric = VecSimMetric_L2,
                              .nLists = 1,
                              .pqBits = 8,
                              .pqDim = 0,
                              .codebookKind = RaftIVFPQ_PerSubspace,
                              .kmeans_nIters = 20,
                              .kmeans_trainsetFraction = 0.5,
                              .nProbes = 20,
                              .lutType = CUDAType_R_32F,
                              .internalDistanceType = CUDAType_R_32F,
                              .preferredShmemCarveout = 1.0};
    return params;
}

RaftIVFFlatParams createDefaultFlatParams(size_t dim)
{
    RaftIVFFlatParams params = {.dim = dim,
                                .metric = VecSimMetric_L2,
                                .nLists = 1,
                                .kmeans_nIters = 20,
                                .kmeans_trainsetFraction = 0.5,
                                .nProbes = 20};
    return params;
}

using DataTypeSetFloat =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>>;

TYPED_TEST_SUITE(RaftIvfTest, DataTypeSetFloat);

TYPED_TEST(RaftIvfTest, RaftIVFFlat_vector_add_test) {
    size_t dim = 4;

    RaftIVFFlatParams params = createDefaultFlatParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFPQ_vector_add_test) {
    size_t dim = 4;

    RaftIVFPQParams params = createDefaultPQParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    GenerateAndAddVector<TEST_DATA_T>(index, dim, 1);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFFlat_add_sanity_test) {
    size_t dim = 4;

    RaftIVFFlatParams params = createDefaultFlatParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)0;
        b[i] = (TEST_DATA_T)1;
        c[i] = (TEST_DATA_T)2;
        d[i] = (TEST_DATA_T)3;
    }

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add first vector. Check it was inserted by searching it.
    VecSimIndex_AddVector(index, a, 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    resultQuery = index->topKQuery(a, 1, &queryParams);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 0);
    ASSERT_EQ(currentResult->score, 0);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);

    // Add second and third vector. Check the topK distances
    VecSimIndex_AddVector(index, b, 1);
    VecSimIndex_AddVector(index, c, 2);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    resultQuery = index->topKQuery(c, 2, &queryParams);
    //VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(resultQuery.results[0].id, 2);
    ASSERT_EQ(resultQuery.results[0].score, 0);
    ASSERT_EQ(resultQuery.results[1].id, 1);
    ASSERT_EQ(resultQuery.results[1].score, 4);
    VecSimQueryResult_Free(resultQuery);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFPQ_add_sanity_test) {
    size_t dim = 4;

    RaftIVFPQParams params = createDefaultPQParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)-10;
        b[i] = (TEST_DATA_T)10;
        c[i] = (TEST_DATA_T)30; 
        d[i] = (TEST_DATA_T)40;
    }

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add first vector. Check it was inserted by searching it.
    VecSimIndex_AddVector(index, a, 0);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    resultQuery = index->topKQuery(a, 1, &queryParams);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 0);
    ASSERT_EQ(currentResult->score, 0);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);

    // Add second and third vector. Check the topK distances
    VecSimIndex_AddVector(index, b, 1);
    VecSimIndex_AddVector(index, c, 9);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    resultQuery = index->topKQuery(b, 3, &queryParams);
    sort_results_by_id(resultQuery);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 0);
    ASSERT_EQ(currentResult->score, 1600);
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 1);
    ASSERT_EQ(currentResult->score, 400); // TODO 0
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 9);
    ASSERT_EQ(currentResult->score, 16);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFFlat_add_unordered_test) {
    size_t dim = 4;

    RaftIVFFlatParams params = createDefaultFlatParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)0;
        b[i] = (TEST_DATA_T)1;
        c[i] = (TEST_DATA_T)2;
        d[i] = (TEST_DATA_T)3;
    }

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add first vector. Check it was inserted by searching it.
    VecSimIndex_AddVector(index, a, 3);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    resultQuery = index->topKQuery(a, 1, &queryParams);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 3);
    ASSERT_EQ(currentResult->score, 0);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);

    // Add second and third vector. Check the topK distances
    VecSimIndex_AddVector(index, b, 4);
    VecSimIndex_AddVector(index, c, 6);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    resultQuery = index->topKQuery(c, 2, &queryParams);
    //VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(resultQuery.results[0].id, 6);
    ASSERT_EQ(resultQuery.results[0].score, 0);
    ASSERT_EQ(resultQuery.results[1].id, 4);
    ASSERT_EQ(resultQuery.results[1].score, 4);
    VecSimQueryResult_Free(resultQuery);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFPQ_add_unordered_test) {
    size_t dim = 4;

    RaftIVFPQParams params = createDefaultPQParams(dim);

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T a[dim], b[dim], c[dim], d[dim];
    for (size_t i = 0; i < dim; i++) {
        a[i] = (TEST_DATA_T)0;
        b[i] = (TEST_DATA_T)1;
        c[i] = (TEST_DATA_T)3; 
        d[i] = (TEST_DATA_T)4;
    }

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add first vector. Check it was inserted by searching it.
    VecSimIndex_AddVector(index, a, 3);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    resultQuery = index->topKQuery(a, 1, &queryParams);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 3);
    ASSERT_EQ(currentResult->score, 0);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);

    // Add second and third vector. Check the topK distances
    VecSimIndex_AddVector(index, b, 4);
    VecSimIndex_AddVector(index, c, 6);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 3);
    resultQuery = index->topKQuery(a, 3, &queryParams);
    sort_results_by_id(resultQuery);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 3);
    ASSERT_EQ(currentResult->score, 4);
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 4);
    ASSERT_EQ(currentResult->score, 4); // TODO 0?
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 6);
    ASSERT_EQ(currentResult->score, 16);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);
    VecSimIndex_Free(index);
}

TYPED_TEST(RaftIvfTest, RaftIVFFlat_batch_add_test) {
    const size_t dim = 4;
    const size_t batch_size = 5;

    RaftIVFFlatParams params = createDefaultFlatParams(dim);
    params.nLists = 3;

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T vectors[dim * batch_size];
    size_t labels[batch_size];
    for (size_t vector_id = 0; vector_id < batch_size; vector_id++) { 
        for (size_t i = 0; i < dim; i++) {
            vectors[vector_id * dim + i] = (TEST_DATA_T)(vector_id * vector_id);
        }
        labels[vector_id] = vector_id;
    }
    // Vectors: {{0, 0, 0, 0}, {1, 1, 1, 1}, {4, 4, 4, 4}, {9, 9, 9, 9}, ...}
    auto flat_index = this->CastToInterface(index);

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add 4 vectors. Check it was inserted by searching it.
    flat_index->addVectorBatch(vectors, labels, 4);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    TEST_DATA_T search_vector[dim] = {6, 6, 6, 6};
    resultQuery = index->topKQuery(search_vector, 2, &queryParams);
    sort_results_by_score(resultQuery);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 2);
    ASSERT_EQ(currentResult->score, 16);
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 3);
    ASSERT_EQ(currentResult->score, 36);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);
}

TYPED_TEST(RaftIvfTest, RaftIVFPQ_batch_add_test) {
    const size_t dim = 4;
    const size_t batch_size = 5;

    RaftIVFPQParams params = createDefaultPQParams(dim);
    params.nLists = 3;

    VecSimIndex *index = this->CreateNewIndex(params);
    VecSimQueryParams queryParams = {.batchSize = 1};

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    TEST_DATA_T vectors[dim * batch_size];
    size_t labels[batch_size];
    for (size_t vector_id = 0; vector_id < batch_size; vector_id++) { 
        for (size_t i = 0; i < dim; i++) {
            vectors[vector_id * dim + i] = (TEST_DATA_T)(vector_id * vector_id);
        }
        labels[vector_id] = vector_id;
    }
    // Vectors: {{0, 0, 0, 0}, {1, 1, 1, 1}, {4, 4, 4, 4}, {9, 9, 9, 9}, ...}
    auto pq_index = this->CastToInterface(index);

    VecSimQueryResult_List resultQuery;
    VecSimQueryResult_Iterator *it = nullptr;
    // Add 4 vectors. Check it was inserted by searching it.
    pq_index->addVectorBatch(vectors, labels, 4);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 4);

    TEST_DATA_T search_vector[dim] = {6, 6, 6, 6};
    resultQuery = index->topKQuery(search_vector, 2, &queryParams);
    sort_results_by_id(resultQuery);
    it = VecSimQueryResult_List_GetIterator(resultQuery);
    VecSimQueryResult *currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 2);
    ASSERT_EQ(currentResult->score, 16);
    currentResult = VecSimQueryResult_IteratorNext(it);
    ASSERT_EQ(currentResult->id, 3);
    ASSERT_EQ(currentResult->score, 36);
    ASSERT_EQ(VecSimQueryResult_IteratorNext(it), nullptr);
    VecSimQueryResult_Free(resultQuery);
    VecSimQueryResult_IteratorFree(it);
}

