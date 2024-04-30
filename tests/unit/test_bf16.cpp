#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "test_utils.h"
#include "mock_thread_pool.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/types/bfloat16.h"

using bfloat16 = vecsim_types::bfloat16;

class BF16Test : public ::testing::Test {
protected:
    // template <typename params_t>
    // void SetUp(params_t &params, bool is_multi = false) {
    //     index = test_utils::CreateNewIndex(params, VecSimType_BFLOAT16, is_multi);
    //     dim = params.dim;
    // }
    virtual void SetUp(HNSWParams &params, bool is_multi = false) {
        FAIL() << "BF16Test::SetUp(HNSWParams) this method should be overriden";
    }

    virtual void SetUp(BFParams &params, bool is_multi = false) {
        FAIL() << "BF16Test::SetUp(BFParams) this method should be overriden";
    }

    virtual void TearDown() { VecSimIndex_Free(index); }

    virtual const void *GetDataByInternalId(idType id) = 0;

    template <typename algo_t>
    algo_t *CastIndex() {
        return dynamic_cast<algo_t *>(index);
    }

    template <typename algo_t>
    algo_t *CastIndex(VecSimIndex *vecsim_index) {
        return dynamic_cast<algo_t *>(vecsim_index);
    }

    void GenerateVector(bfloat16 *out_vec, float initial_value = 0.5f, float step = 1.0f) {
        for (size_t i = 0; i < dim; i++) {
            out_vec[i] = vecsim_types::float_to_bf16(initial_value + step * i);
        }
    }

    int GenerateAndAddVector(size_t id, float initial_value = 0.5f, float step = 1.0f) {
        bfloat16 v[dim];
        GenerateVector(v, initial_value, step);
        return VecSimIndex_AddVector(index, v, id);
    }

    template <typename params_t>
    void create_index_test(params_t index_params);
    template <typename params_t>
    void search_by_id_test(params_t index_params);
    template <typename params_t>
    void search_by_score_test(params_t index_params);
    template <typename params_t>
    void search_empty_index_test(params_t index_params);
    template <typename params_t>
    void test_info_iterator(VecSimMetric metric);

    VecSimIndex *index;
    size_t dim;
};

class BF16HNSWTest : public BF16Test {
protected:
    virtual void SetUp(HNSWParams &params, bool is_multi = false) {
        index = test_utils::CreateNewIndex(params, VecSimType_BFLOAT16, is_multi);
        dim = params.dim;
    }

    virtual const void *GetDataByInternalId(idType id) {
        return CastIndex<HNSWIndex_Single<bfloat16, float>>()->getDataByInternalId(id);
    }
};

class BF16BruteForceTest : public BF16Test {
protected:
    virtual void SetUp(BFParams &params, bool is_multi = false) {
        index = test_utils::CreateNewIndex(params, VecSimType_BFLOAT16, is_multi);
        dim = params.dim;
    }

    virtual const void *GetDataByInternalId(idType id) {
        return CastIndex<BruteForceIndex_Single<bfloat16, float>>()->getDataByInternalId(id);
    }
};

class BF16TieredTest : public BF16Test {
protected:
    TieredIndexParams generate_tiered_params(HNSWParams &hnsw_params) {
        vecsim_hnsw_params = CreateParams(hnsw_params);
        TieredIndexParams tiered_params = {.jobQueue = &mock_thread_pool.jobQ,
                                           .jobQueueCtx = mock_thread_pool.ctx,
                                           .submitCb = tieredIndexMock::submit_callback,
                                           .flatBufferLimit = SIZE_MAX,
                                           .primaryIndexParams = &vecsim_hnsw_params};
        return tiered_params;
    }

    virtual void SetUp(HNSWParams &hnsw_params, bool is_multi = false) override {
        hnsw_params.type = VecSimType_BFLOAT16;
        TieredIndexParams tiered_params = generate_tiered_params(hnsw_params);
        VecSimParams params = CreateParams(tiered_params);
        index = VecSimIndex_New(&params);
        dim = hnsw_params.dim;

        // Set the created tiered index in the index external context.
        mock_thread_pool.ctx->index_strong_ref.reset(index);
    }

    virtual void TearDown() override {}

    virtual const void *GetDataByInternalId(idType id) {
        return CastIndex<BruteForceIndex<bfloat16, float>>(GetBruteForce())
            ->getDataByInternalId(id);
    }

    HNSWIndex<bfloat16, float> *GetHNSW() {
        auto tiered_index = dynamic_cast<TieredHNSWIndex<bfloat16, float> *>(index);
        return tiered_index->getHNSWIndex();
    }

    VecSimIndexAbstract<bfloat16, float> *GetBruteForce() {
        auto tiered_index = dynamic_cast<TieredHNSWIndex<bfloat16, float> *>(index);
        return tiered_index->getFlatBufferIndex();
    }

    VecSimParams vecsim_hnsw_params;
    tieredIndexMock mock_thread_pool;
};
/* ---------------------------- Create index tests ---------------------------- */

template <typename params_t>
void BF16Test::create_index_test(params_t index_params) {
    SetUp(index_params);
    float initial_value = 0.5f;
    float step = 1.0f;

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    bfloat16 vector[dim];
    this->GenerateVector(vector, initial_value, step);
    VecSimIndex_AddVector(index, vector, 0);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    ASSERT_EQ(index->getDistanceFrom_Unsafe(0, vector), 0);

    const void *v = this->GetDataByInternalId(0);

    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(((bfloat16 *)v)[i]),
                  initial_value + step * float(i));
    }
}

TEST_F(BF16HNSWTest, createIndex) {
    HNSWParams params = {.dim = 40, .initialCapacity = 200, .M = 16, .efConstruction = 200};
    create_index_test(params);
    ASSERT_EQ(index->basicInfo().type, VecSimType_BFLOAT16);
    ASSERT_EQ(index->basicInfo().algo, VecSimAlgo_HNSWLIB);
}

TEST_F(BF16BruteForceTest, createIndex) {
    BFParams params = {.dim = 40, .initialCapacity = 200};
    create_index_test(params);
    ASSERT_EQ(index->basicInfo().type, VecSimType_BFLOAT16);
    ASSERT_EQ(index->basicInfo().algo, VecSimAlgo_BF);
}

TEST_F(BF16TieredTest, createIndex) {
    HNSWParams params = {.dim = 40, .initialCapacity = 200, .M = 16, .efConstruction = 200};
    create_index_test(params);
    ASSERT_EQ(index->basicInfo().type, VecSimType_BFLOAT16);
    ASSERT_EQ(index->basicInfo().isTiered, true);
}

TEST_F(BF16TieredTest, createIndexTiered) {
    HNSWParams hnsw_params = {.dim = 40, .initialCapacity = 200, .M = 16};

    SetUp(hnsw_params);
    ASSERT_EQ(index->basicInfo().type, VecSimType_BFLOAT16);
    ASSERT_EQ(index->basicInfo().isTiered, true);

    // Add a vector to the flat index.
    bfloat16 vector[dim];
    labelType vec_label = 1;
    float initial_value = 0.5f;
    float step = 1.0f;
    this->GenerateVector(vector, initial_value, step);
    VecSimIndex_AddVector(index, vector, vec_label);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    ASSERT_EQ(index->getDistanceFrom_Unsafe(vec_label, vector), 0);

    const void *v =
        CastIndex<BruteForceIndex<bfloat16, float>>(GetBruteForce())->getDataByInternalId(0);

    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(vecsim_types::bfloat16_to_float32(((bfloat16 *)v)[i]),
                  initial_value + step * float(i));
    }
}

/* ---------------------------- Size Estimation tests ---------------------------- */

TEST_F(BF16HNSWTest, testSizeEstimation) {
    size_t n = 200;
    size_t bs = 256;
    size_t M = 64;

    // Initial capacity is rounded up to the block size.
    size_t extra_cap = n % bs == 0 ? 0 : bs - n % bs;

    HNSWParams params = {.dim = 256, .initialCapacity = n, .blockSize = bs, .M = M};
    SetUp(params);

    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    estimation +=
        (this->CastIndex<HNSWIndex_Single<bfloat16, float>>()->labelLookup.bucket_count() -
         (n + extra_cap)) *
        sizeof(size_t);

    ASSERT_EQ(estimation, actual);

    // Fill the initial capacity + fill the last block.
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQ(this->GenerateAndAddVector(i), 1);
    }
    idType cur = n;
    while (index->indexSize() % bs != 0) {
        this->GenerateAndAddVector(cur++);
    }

    // Estimate the memory delta of adding a single vector that requires a full new block.
    estimation = EstimateElementSize(params) * bs;
    size_t before = index->getAllocationSize();
    this->GenerateAndAddVector(bs);
    actual = index->getAllocationSize() - before;

    // We check that the actual size is within 1% of the estimation.
    ASSERT_GE(estimation, actual * 0.99);
    ASSERT_LE(estimation, actual * 1.01);
}

TEST_F(BF16HNSWTest, testSizeEstimation_No_InitialCapacity) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    HNSWParams params = {.dim = dim, .initialCapacity = n, .blockSize = bs};
    SetUp(params);

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
}

TEST_F(BF16BruteForceTest, testSizeEstimation) {
    size_t dim = 128;
    size_t n = 0;
    size_t bs = DEFAULT_BLOCK_SIZE;

    BFParams params = {.dim = dim, .initialCapacity = n, .blockSize = bs};
    SetUp(params);

    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);

    estimation = EstimateElementSize(params) * bs;

    ASSERT_EQ(this->GenerateAndAddVector(0), 1);

    actual = index->getAllocationSize() - actual; // get the delta
    ASSERT_GE(estimation * 1.01, actual);
    ASSERT_LE(estimation * 0.99, actual);
}

TEST_F(BF16BruteForceTest, testSizeEstimation_No_InitialCapacity) {
    size_t dim = 128;
    size_t n = 100;
    size_t bs = DEFAULT_BLOCK_SIZE;

    BFParams params = {
        .dim = dim, .metric = VecSimMetric_Cosine, .initialCapacity = n, .blockSize = bs};
    SetUp(params);

    // EstimateInitialSize is called after CreateNewIndex because params struct is
    // changed in CreateNewIndex.
    size_t estimation = EstimateInitialSize(params);

    size_t actual = index->getAllocationSize();
    ASSERT_EQ(estimation, actual);
}

TEST_F(BF16TieredTest, testSizeEstimation) {
    size_t n = DEFAULT_BLOCK_SIZE;
    size_t M = 32;
    size_t bs = DEFAULT_BLOCK_SIZE;

    HNSWParams hnsw_params = {.dim = 128, .initialCapacity = n, .M = M};
    SetUp(hnsw_params);
    TieredIndexParams tiered_params = generate_tiered_params(hnsw_params);
    VecSimParams params = CreateParams(tiered_params);

    // auto allocator = index->getAllocator();
    size_t initial_size_estimation = VecSimIndex_EstimateInitialSize(&params);

    // labels_lookup hash table has additional memory, since STL implementation chooses "an
    // appropriate prime number" higher than n as the number of allocated buckets (for n=1000, 1031
    // buckets are created)
    auto hnsw_index = GetHNSW();
    // if (isMulti == false) {
    auto hnsw = CastIndex<HNSWIndex_Single<bfloat16, float>>(hnsw_index);
    initial_size_estimation += (hnsw->labelLookup.bucket_count() - n) * sizeof(size_t);
    // } else {
    //     // if its a multi value index cast to HNSW_Multi
    //     auto hnsw = reinterpret_cast<HNSWIndex_Multi<TEST_DATA_T, TEST_DIST_T> *>(hnsw_index);
    //     initial_size_estimation += (hnsw->labelLookup.bucket_count() - n) * sizeof(size_t);
    // }

    ASSERT_EQ(initial_size_estimation, index->getAllocationSize());

    // Add vectors up to initial capacity (initial capacity == block size).
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector(i, i, 0);
        mock_thread_pool.thread_iteration();
    }

    // Estimate memory delta for filling up the first block and adding another block.
    size_t estimation = VecSimIndex_EstimateElementSize(&params) * bs;

    size_t before = index->getAllocationSize();
    GenerateAndAddVector(bs + n, bs + n, 0);
    mock_thread_pool.thread_iteration();
    size_t actual = index->getAllocationSize() - before;

    // Flat index should be empty, hence the index size includes only hnsw size.
    ASSERT_EQ(index->indexSize(), hnsw_index->indexSize());
    ASSERT_EQ(index->indexCapacity(), hnsw_index->indexCapacity());
    // We added n + 1 vectors
    ASSERT_EQ(index->indexSize(), n + 1);
    // We should have 2 blocks now
    ASSERT_EQ(index->indexCapacity(), 2 * bs);

    // We check that the actual size is within 1% of the estimation.
    ASSERT_GE(estimation, actual * 0.99);
    ASSERT_LE(estimation, actual * 1.01);
}

/* ---------------------------- Vector search tests ---------------------------- */

template <typename params_t>
void BF16Test::search_by_id_test(params_t index_params) {
    SetUp(index_params);

    size_t k = 11;
    size_t n = 100;

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(i, i, 0); // {i, i, i, i}
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    bfloat16 query[dim];
    GenerateVector(query, 50, 0);

    // Vectors values are equal to the id, so the 11 closest vectors are 45, 46...50
    // (closest), 51...55
    static size_t expected_res_order[] = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, expected_res_order[index]);    // results are sorted by ID
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };

    runTopKSearchTest(index, query, k, verify_res, nullptr, BY_ID);
}

TEST_F(BF16HNSWTest, searchByID) {
    HNSWParams params = {.dim = 4, .initialCapacity = 200, .M = 16, .efConstruction = 200};
    search_by_id_test(params);
}

TEST_F(BF16BruteForceTest, searchByID) {
    BFParams params = {.dim = 4, .initialCapacity = 200};
    search_by_id_test(params);
}

TEST_F(BF16TieredTest, searchByID) {
    HNSWParams params = {.dim = 4, .initialCapacity = 200, .M = 16, .efConstruction = 200};
    search_by_id_test(params);
}

template <typename params_t>
void BF16Test::search_by_score_test(params_t index_params) {
    SetUp(index_params);

    size_t k = 11;
    size_t n = 100;

    for (size_t i = 0; i < n; i++) {
        this->GenerateAndAddVector(i, i, 0); // {i, i, i, i}
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    bfloat16 query[dim];
    GenerateVector(query, 50, 0);
    // Vectors values are equal to the id, so the 11 closest vectors are
    // 45, 46...50 (closest), 51...55
    static size_t expected_res_order[] = {50, 49, 51, 48, 52, 47, 53, 46, 54, 45, 55};
    auto verify_res = [&](size_t id, double score, size_t index) {
        ASSERT_EQ(id, expected_res_order[index]);
        ASSERT_EQ(score, 4 * (50 - id) * (50 - id)); // L2 distance
    };

    runTopKSearchTest(index, query, k, verify_res);
}

TEST_F(BF16HNSWTest, searchByScore) {
    HNSWParams params = {.dim = 4, .initialCapacity = 200, .M = 16, .efConstruction = 200};
    search_by_score_test(params);
}

TEST_F(BF16BruteForceTest, searchByScore) {
    BFParams params = {.dim = 4, .initialCapacity = 200};
    search_by_score_test(params);
}

TEST_F(BF16TieredTest, searchByScore) {
    HNSWParams params = {.dim = 4, .initialCapacity = 200, .M = 16, .efConstruction = 200};
    search_by_score_test(params);
}

template <typename params_t>
void BF16Test::search_empty_index_test(params_t params) {
    size_t n = 100;
    size_t k = 11;

    SetUp(params);
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    bfloat16 query[dim];
    GenerateVector(query, 50, 0);

    // We do not expect any results.
    VecSimQueryReply *res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Iterator *it = VecSimQueryReply_GetIterator(res);
    ASSERT_EQ(VecSimQueryReply_IteratorNext(it), nullptr);
    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(res);

    res = VecSimIndex_RangeQuery(index, query, 1.0, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);

    // Add some vectors and remove them all from index, so it will be empty again.
    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector(i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_DeleteVector(index, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    // Again - we do not expect any results.
    res = VecSimIndex_TopKQuery(index, query, k, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    it = VecSimQueryReply_GetIterator(res);
    ASSERT_EQ(VecSimQueryReply_IteratorNext(it), nullptr);
    VecSimQueryReply_IteratorFree(it);
    VecSimQueryReply_Free(res);

    res = VecSimIndex_RangeQuery(index, query, 1.0, NULL, BY_SCORE);
    ASSERT_EQ(VecSimQueryReply_Len(res), 0);
    VecSimQueryReply_Free(res);
}

TEST_F(BF16HNSWTest, SearchEmptyIndex) {
    HNSWParams params = {.dim = 4, .initialCapacity = 0};
    search_empty_index_test(params);
}

TEST_F(BF16BruteForceTest, SearchEmptyIndex) {
    BFParams params = {.dim = 4, .initialCapacity = 0};
    search_empty_index_test(params);
}

TEST_F(BF16TieredTest, SearchEmptyIndex) {
    HNSWParams params = {.dim = 4, .initialCapacity = 0};
    search_empty_index_test(params);
}

// float calc_dist_l2(const bfloat16 *vec1, const bfloat16 *vec2, size_t dim) {
//     float dist = 0;
//     for (size_t i = 0; i < dim; i++) {
//         float diff =
//             vecsim_types::bfloat16_to_float32(vec1[i]) -
//             vecsim_types::bfloat16_to_float32(vec2[i]);
//         dist += diff * diff;
//     }
//     return dist;
// }

/* ---------------------------- Info tests ---------------------------- */
TEST_F(BF16HNSWTest, Info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args
    HNSWParams params = {.dim = d, .metric = VecSimMetric_L2, .initialCapacity = n};
    SetUp(params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_HNSWLIB);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
    // Default args.
    ASSERT_FALSE(info.commonInfo.basicInfo.isMulti);
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.hnswInfo.M, HNSW_DEFAULT_M);
    ASSERT_EQ(info.hnswInfo.efConstruction, HNSW_DEFAULT_EF_C);
    ASSERT_EQ(info.hnswInfo.efRuntime, HNSW_DEFAULT_EF_RT);
    ASSERT_DOUBLE_EQ(info.hnswInfo.epsilon, HNSW_DEFAULT_EPSILON);
    ASSERT_EQ(info.commonInfo.basicInfo.type, VecSimType_BFLOAT16);

    // Validate that Static info returns the right restricted info as well.
    VecSimIndexBasicInfo s_info = VecSimIndex_BasicInfo(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, s_info.algo);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, s_info.dim);
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, s_info.blockSize);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isMulti, s_info.isMulti);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isTiered, s_info.isTiered);
}

TEST_F(BF16BruteForceTest, Info) {
    size_t n = 100;
    size_t d = 128;

    // Build with default args.

    BFParams params = {.dim = d, .metric = VecSimMetric_L2, .initialCapacity = n};
    SetUp(params);

    VecSimIndexInfo info = VecSimIndex_Info(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, VecSimAlgo_BF);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, d);
    ASSERT_FALSE(info.commonInfo.basicInfo.isMulti);
    ASSERT_EQ(info.commonInfo.basicInfo.type, VecSimType_BFLOAT16);

    // Default args.
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, DEFAULT_BLOCK_SIZE);
    ASSERT_EQ(info.commonInfo.indexSize, 0);

    // Validate that Static info returns the right restricted info as well.
    VecSimIndexBasicInfo s_info = VecSimIndex_BasicInfo(index);
    ASSERT_EQ(info.commonInfo.basicInfo.algo, s_info.algo);
    ASSERT_EQ(info.commonInfo.basicInfo.dim, s_info.dim);
    ASSERT_EQ(info.commonInfo.basicInfo.blockSize, s_info.blockSize);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isMulti, s_info.isMulti);
    ASSERT_EQ(info.commonInfo.basicInfo.type, s_info.type);
    ASSERT_EQ(info.commonInfo.basicInfo.isTiered, s_info.isTiered);
}

template <typename params_t>
void BF16Test::test_info_iterator(VecSimMetric metric) {
    size_t n = 100;
    size_t d = 128;
    params_t params = {.dim = d, .metric = metric, .initialCapacity = n};
    SetUp(params);
    VecSimIndexInfo info = VecSimIndex_Info(index);
    VecSimInfoIterator *infoIter = VecSimIndex_InfoIterator(index);
    VecSimAlgo algo = info.commonInfo.basicInfo.algo;
    if (algo == VecSimAlgo_HNSWLIB) {
        compareHNSWIndexInfoToIterator(info, infoIter);
    } else if (algo == VecSimAlgo_BF) {
        compareFlatIndexInfoToIterator(info, infoIter);
    }
    VecSimInfoIterator_Free(infoIter);
}

TEST_F(BF16BruteForceTest, InfoIteratorCosine) {
    test_info_iterator<BFParams>(VecSimMetric_Cosine);
}
TEST_F(BF16BruteForceTest, InfoIteratorIP) { test_info_iterator<BFParams>(VecSimMetric_IP); }
TEST_F(BF16BruteForceTest, InfoIteratorL2) { test_info_iterator<BFParams>(VecSimMetric_L2); }
TEST_F(BF16HNSWTest, InfoIteratorCosine) { test_info_iterator<HNSWParams>(VecSimMetric_Cosine); }
TEST_F(BF16HNSWTest, InfoIteratorIP) { test_info_iterator<HNSWParams>(VecSimMetric_IP); }
TEST_F(BF16HNSWTest, InfoIteratorL2) { test_info_iterator<HNSWParams>(VecSimMetric_L2); }

// TEST_F(BF16Test, brute_force_vector_search_test_l2) {
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

// TEST_F(BF16Test, hnsw_vector_search_test) {
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
