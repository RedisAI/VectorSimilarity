#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "test_utils.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/query_result_definitions.h"
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>
#include "VecSim/types/bfloat16.h"

using bfloat16 = vecsim_types::bfloat16;

template <typename index_type_t>
class BF16Test : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *CreateNewIndex(HNSWParams &params, bool is_multi = false) {
        params.metric = VecSimMetric_L2;
        VecSimIndex *index =
            test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
        assert(index->basicInfo().type == VecSimType_BFLOAT16);
        assert(index->basicInfo().algo == VecSimAlgo_HNSWLIB);

        return index;
    }

    VecSimIndex *CreateNewIndex(BFParams &params, bool is_multi = false) {
        params.metric = VecSimMetric_L2;
        VecSimIndex *index =
            test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
        assert(index->basicInfo().type == VecSimType_BFLOAT16);
        assert(index->basicInfo().algo == VecSimAlgo_BF);

        return index;
    }

    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return dynamic_cast<HNSWIndex<data_t, dist_t> *>(index);
    }

    HNSWIndex_Single<data_t, dist_t> *CastToHNSW_Single(VecSimIndex *index) {
        return dynamic_cast<HNSWIndex_Single<data_t, dist_t> *>(index);
    }

    BruteForceIndex<data_t, dist_t> *CastToBruteForce(VecSimIndex *index) {
        return dynamic_cast<BruteForceIndex<data_t, dist_t> *>(index);
    }

    void FloatVecToBfloat16Vec(float *org_vec, bfloat16 *bf_vec, size_t dim) {
        for (size_t i = 0; i < dim; i++) {
            bf_vec[i] = vecsim_types::float_to_bf16(org_vec[i]);
        }
    }

    int GenerateAndAddVector(VecSimIndex *index, size_t dim, size_t id, float initial_value = 0.5f,
                             float step = 1.0f) {
        bfloat16 v[dim];
        for (size_t i = 0; i < dim; i++) {
            v[i] = vecsim_types::float_to_bf16(initial_value + step * i);
        }
        return VecSimIndex_AddVector(index, v, id);
    }
};

using DataTypeBF16 = ::testing::Types<IndexType<VecSimType_BFLOAT16, bfloat16, float>>;

TYPED_TEST_SUITE(BF16Test, DataTypeBF16);

TYPED_TEST(BF16Test, create_hnsw_index) {
    size_t dim = 40;
    float initial_value = 0.5f;
    float step = 1.0f;

    HNSWParams params = {.dim = dim, .initialCapacity = 200, .M = 16, .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    this->GenerateAndAddVector(index, dim, 0, initial_value, step);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    const void *v = this->CastToHNSW_Single(index)->getDataByInternalId(0);

    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(((bfloat16 *)v)[i]),
                  initial_value + step * float(i));
    }
    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, create_bf_index) {
    size_t dim = 40;
    float initial_value = 0.5f;
    float step = 1.0f;

    BFParams params = {.dim = dim, .initialCapacity = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    this->GenerateAndAddVector(index, dim, 0, initial_value, step);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);

    const void *v = this->CastToBruteForce(index)->getDataByInternalId(0);

    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(((bfloat16 *)v)[i]),
                  initial_value + step * float(i));
    }

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, testSizeEstimationHNSW) {
    size_t dim = 256;
    size_t n = 200;
    size_t bs = 256;
    size_t M = 64;

    // Initial capacity is rounded up to the block size.
    size_t extra_cap = n % bs == 0 ? 0 : bs - n % bs;

    HNSWParams params = {.dim = dim, .initialCapacity = n, .blockSize = bs, .M = M};

    VecSimIndex *index = this->CreateNewIndex(params);

    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    estimation += (this->CastToHNSW_Single(index)->labelLookup.bucket_count() - (n + extra_cap)) *
                  sizeof(size_t);

    ASSERT_EQ(estimation, actual);

    // Fill the initial capacity + fill the last block.
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(this->GenerateAndAddVector(index, dim, i), 1);
    }
    idType cur = n;
    while (index->indexSize() % bs != 0) {
        this->GenerateAndAddVector(index, dim, cur++);
    }

    // Estimate the memory delta of adding a single vector that requires a full new block.
    estimation = EstimateElementSize(params) * bs;
    size_t before = index->getAllocationSize();
    this->GenerateAndAddVector(index, dim, bs);
    actual = index->getAllocationSize() - before;

    // We check that the actual size is within 1% of the estimation.
    ASSERT_GE(estimation, actual * 0.99);
    ASSERT_LE(estimation, actual * 1.01);

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, testSizeEstimation_No_InitialCapacityHNSW) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    HNSWParams params = {.dim = dim, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();

    // labels_lookup and element_levels containers are not allocated at all in some platforms,
    // when initial capacity is zero, while in other platforms labels_lookup is allocated with a
    // single bucket. This, we get the following range in which we expect the initial memory to be
    // in.
    ASSERT_GE(actual, estimation);
    ASSERT_LE(actual, estimation + sizeof(size_t) + 2 * sizeof(size_t));

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, testSizeEstimationBF) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    BFParams params = {.dim = dim, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    estimation = EstimateElementSize(params) * bs;

    ASSERT_EQ(this->GenerateAndAddVector(index, dim, 0), 1);

    actual = index->getAllocationSize() - actual; // get the delta
    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, testSizeEstimation_No_InitialCapacityBF) {
    size_t dim = 128;
    size_t n = 100;
    size_t bs = DEFAULT_BLOCK_SIZE;

    BFParams params = {
        .dim = dim, .metric = VecSimMetric_Cosine, .initialCapacity = n, .blockSize = bs};

    VecSimIndex *index = this->CreateNewIndex(params);
    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, vector_search_by_id_test_brute_force) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    BFParams params = {.dim = dim, .initialCapacity = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i, 0); // {i, i, i, i}
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    bfloat16 bf_query[dim];
    this->FloatVecToBfloat16Vec(query, bf_query, dim);
    // Vectors values are equal to the id, so the 11 closest vectors are 45, 46...50
    // (closest), 51...55
    static size_t expected_res_order[] = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, expected_res_order[index]);    // results are sorted by ID
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, vector_search_by_score_test_brute_force) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    BFParams params = {.dim = dim, .initialCapacity = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i, 0); // {i, i, i, i}
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    bfloat16 bf_query[dim];
    this->FloatVecToBfloat16Vec(query, bf_query, dim);
    // Vectors values are equal to the id, so the 11 closest vectors are
    // 45, 46...50 (closest), 51...55
    static size_t expected_res_order[] = {50, 49, 51, 48, 52, 47, 53, 46, 54, 45, 55};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, expected_res_order[index]);
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };

    runTopKSearchTest(index, bf_query, k, verify_res);
    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, vector_search_by_id_test_hnsw) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i, 0); // {i, i, i, i}
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    bfloat16 bf_query[dim];
    this->FloatVecToBfloat16Vec(query, bf_query, dim);
    // Vectors values are equal to the id, so the 11 closest vectors are
    // 45, 46...50 (closest), 51...55
    static size_t expected_res_order[] = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, expected_res_order[index]);    // results are sorted by ID
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };
    runTopKSearchTest(index, bf_query, k, verify_res, nullptr, BY_ID);

    VecSimIndex_Free(index);
}

TYPED_TEST(BF16Test, vector_search_by_score_test_hnsw) {
    size_t n = 100;
    size_t k = 11;
    size_t dim = 4;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = 200,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(index, dim, i, i, 0); // {i, i, i, i}
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    float query[] = {50, 50, 50, 50};
    bfloat16 bf_query[dim];
    this->FloatVecToBfloat16Vec(query, bf_query, dim);
    // Vectors values are equal to the id, so the 11 closest vectors are
    // 45, 46...50 (closest), 51...55
    static size_t expected_res_order[] = {50, 49, 51, 48, 52, 47, 53, 46, 54, 45, 55};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, expected_res_order[index]);
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };

    runTopKSearchTest(index, bf_query, k, verify_res);
    VecSimIndex_Free(index);
}

float calc_dist_l2(const bfloat16 *vec1, const bfloat16 *vec2, size_t dim) {
    float dist = 0;
    for (size_t i = 0; i < dim; i++) {
        float diff =
            vecsim_types::bfloat16_to_float32(vec1[i]) - vecsim_types::bfloat16_to_float32(vec2[i]);
        dist += diff * diff;
    }
    return dist;
}
// TYPED_TEST(BF16Test, brute_force_vector_search_test_l2) {
//     size_t dim = 3;
//     size_t n = 1;
//     size_t k = 1;

//     BFParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = 55};

//     VecSimIndex *index = this->CreateNewIndex(params);

//     bfloat16 v[dim];
//     for (size_t i = 0; i < dim; i++) {
//         v[i] = vecsim_types::float_to_bf16(0.5 + i * 0.25f);
//         std::cout << vecsim_types::bfloat16_to_float32(v[i]) << std::endl;
//     }
//     VecSimIndex_AddVector(index, v, 0);
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     bfloat16 query[dim];
//     for (size_t i = 0; i < dim; i++) {
//         query[i] = vecsim_types::float_to_bf16(i * 0.25f);
//         std::cout << vecsim_types::bfloat16_to_float32(query[i]) << std::endl;
//     }
//     VecSimQueryReply *res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_ID);
//     ASSERT_EQ(VecSimQueryReply_Len(res), k);
//     VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(res);
//     int res_ind = 0;
//     while (VecSimQueryReply_IteratorHasNext(iterator)) {
//         VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
//         int id = (int)VecSimQueryResult_GetId(item);
//         double score = VecSimQueryResult_GetScore(item);
//         ASSERT_EQ(id, 0);
//         ASSERT_EQ(score, 0.02);
//     }
//     VecSimQueryReply_IteratorFree(iterator);
//     VecSimIndex_Free(index);
// }

// TYPED_TEST(BF16Test, hnsw_vector_search_test) {
//     size_t dim = 3;
//     size_t n = 1;
//     size_t k = 1;

//     HNSWParams params = {.dim = dim, .metric = VecSimMetric_L2, .initialCapacity = 55};

//     VecSimIndex *index = this->CreateNewIndex(params);

//     bfloat16 v[dim];
//     for (size_t i = 0; i < dim; i++) {
//         v[i] = vecsim_types::float_to_bf16(0.5 + i * 0.25f);
//         std::cout << vecsim_types::bfloat16_to_float32(v[i]) << std::endl;
//     }
//     VecSimIndex_AddVector(index, v, 0);
//     ASSERT_EQ(VecSimIndex_IndexSize(index), n);

//     bfloat16 query[dim];
//     for (size_t i = 0; i < dim; i++) {
//         query[i] = vecsim_types::float_to_bf16(i * 0.25f);
//         std::cout << vecsim_types::bfloat16_to_float32(query[i]) << std::endl;
//     }
//     VecSimQueryReply *res = VecSimIndex_TopKQuery(index, query, k, nullptr, BY_ID);
//     ASSERT_EQ(VecSimQueryReply_Len(res), k);
//     VecSimQueryReply_Iterator *iterator = VecSimQueryReply_GetIterator(res);
//     int res_ind = 0;
//     while (VecSimQueryReply_IteratorHasNext(iterator)) {
//         VecSimQueryResult *item = VecSimQueryReply_IteratorNext(iterator);
//         int id = (int)VecSimQueryResult_GetId(item);
//         double score = VecSimQueryResult_GetScore(item);
//         ASSERT_EQ(id, 0);
//         ASSERT_EQ(score, 0.02);
//     }
//     VecSimQueryReply_IteratorFree(iterator);
//     VecSimIndex_Free(index);
// }
