#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
// #include "VecSim/index_factories/hnsw_factory.h"
#include "tests_utils.h"
#include "unit_test_utils.h"
// #include "VecSim/utils/serializer.h"
#include "mock_thread_pool.h"
// #include "VecSim/query_result_definitions.h"
// #include "VecSim/vec_sim_debug.h"
// #include "VecSim/spaces/L2/L2.h"

class INT8Test : public ::testing::Test {
protected:
    virtual void SetUp(HNSWParams &params) {
        FAIL() << "INT8Test::SetUp(HNSWParams) this method should be overriden";
    }

    virtual void SetUp(BFParams &params) {
        FAIL() << "INT8Test::SetUp(BFParams) this method should be overriden";
    }

    virtual void SetUp(TieredIndexParams &tiered_params) {
        FAIL() << "INT8Test::SetUp(TieredIndexParams) this method should be overriden";
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

    void GenerateVector(int8_t *out_vec) { test_utils::populate_int8_vec(out_vec, dim); }

    int GenerateAndAddVector(size_t id) {
        int8_t v[dim];
        GenerateVector(v);
        return VecSimIndex_AddVector(index, v, id);
    }
    template <typename params_t>
    void create_index_test(params_t index_params);

    VecSimIndex *index;
    size_t dim;
};

class INT8HNSWTest : public INT8Test {
protected:
    virtual void SetUp(HNSWParams &params) override {
        params.type = VecSimType_INT8;
        VecSimParams vecsim_params = CreateParams(params);
        index = VecSimIndex_New(&vecsim_params);
        dim = params.dim;
    }

    virtual const void *GetDataByInternalId(idType id) override {
        return CastIndex<HNSWIndex_Single<int8_t, float>>()->getDataByInternalId(id);
    }
};

class INT8BruteForceTest : public INT8Test {
protected:
    virtual void SetUp(BFParams &params) override {
        params.type = VecSimType_INT8;
        VecSimParams vecsim_params = CreateParams(params);
        index = VecSimIndex_New(&vecsim_params);
        dim = params.dim;
    }

    virtual const void *GetDataByInternalId(idType id) override {
        return CastIndex<BruteForceIndex_Single<int8_t, float>>()->getDataByInternalId(id);
    }
};

class INT8TieredTest : public INT8Test {
protected:
    TieredIndexParams generate_tiered_params(HNSWParams &hnsw_params, size_t swap_job_threshold = 0,
                                             size_t flat_buffer_limit = SIZE_MAX) {
        hnsw_params.type = VecSimType_INT8;
        vecsim_hnsw_params = CreateParams(hnsw_params);
        TieredIndexParams tiered_params = {
            .jobQueue = &mock_thread_pool.jobQ,
            .jobQueueCtx = mock_thread_pool.ctx,
            .submitCb = tieredIndexMock::submit_callback,
            .flatBufferLimit = flat_buffer_limit,
            .primaryIndexParams = &vecsim_hnsw_params,
            .specificParams = {TieredHNSWParams{.swapJobThreshold = swap_job_threshold}}};
        return tiered_params;
    }

    virtual void SetUp(TieredIndexParams &tiered_params) override {
        VecSimParams params = CreateParams(tiered_params);
        index = VecSimIndex_New(&params);
        dim = tiered_params.primaryIndexParams->algoParams.hnswParams.dim;

        // Set the created tiered index in the index external context.
        mock_thread_pool.ctx->index_strong_ref.reset(index);
    }

    virtual void SetUp(HNSWParams &hnsw_params) override {
        TieredIndexParams tiered_params = generate_tiered_params(hnsw_params);
        SetUp(tiered_params);
    }

    virtual void TearDown() override {}

    virtual const void *GetDataByInternalId(idType id) override {
        return CastIndex<BruteForceIndex<int8_t, float>>(CastToBruteForce())
            ->getDataByInternalId(id);
    }

    VecSimIndexAbstract<int8_t, float> *CastToBruteForce() {
        auto tiered_index = dynamic_cast<TieredHNSWIndex<int8_t, float> *>(index);
        return tiered_index->getFlatBufferIndex();
    }

    VecSimParams vecsim_hnsw_params;
    tieredIndexMock mock_thread_pool;
};

/* ---------------------------- Create index tests ---------------------------- */

template <typename params_t>
void INT8Test::create_index_test(params_t index_params) {
    SetUp(index_params);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 0);

    int8_t vector[dim];
    this->GenerateVector(vector);
    VecSimIndex_AddVector(index, vector, 0);

    ASSERT_EQ(VecSimIndex_IndexSize(index), 1);
    ASSERT_EQ(index->getDistanceFrom_Unsafe(0, vector), 0);

    ASSERT_NO_FATAL_FAILURE(
        CompareVectors(static_cast<const int8_t *>(this->GetDataByInternalId(0)), vector, dim));
}

TEST_F(INT8HNSWTest, createIndex) {
    HNSWParams params = {.dim = 40, .M = 16, .efConstruction = 200};
    EXPECT_NO_FATAL_FAILURE(create_index_test(params));
    ASSERT_EQ(index->basicInfo().type, VecSimType_INT8);
    ASSERT_EQ(index->basicInfo().algo, VecSimAlgo_HNSWLIB);
}

TEST_F(INT8BruteForceTest, createIndex) {
    BFParams params = {.dim = 40};
    EXPECT_NO_FATAL_FAILURE(create_index_test(params));
    ASSERT_EQ(index->basicInfo().type, VecSimType_INT8);
    ASSERT_EQ(index->basicInfo().algo, VecSimAlgo_BF);
}

TEST_F(INT8TieredTest, createIndex) {
    HNSWParams params = {.dim = 40, .M = 16, .efConstruction = 200};
    EXPECT_NO_FATAL_FAILURE(create_index_test(params));
    ASSERT_EQ(index->basicInfo().type, VecSimType_INT8);
    ASSERT_EQ(index->basicInfo().isTiered, true);
}
