#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
// #include "VecSim/index_factories/hnsw_factory.h"
#include "tests_utils.h"
#include "test_utils.h"
// #include "VecSim/utils/serializer.h"
// #include "mock_thread_pool.h"
// #include "VecSim/query_result_definitions.h"
// #include "VecSim/types/float16.h"
// #include "VecSim/vec_sim_debug.h"
// #include "VecSim/spaces/L2/L2.h"

class INT8Test : public ::testing::Test {
protected:
    virtual void SetUp(HNSWParams &params) {
        FAIL() << "INT8Test::SetUp(HNSWParams) this method should be overriden";
    }

    virtual void TearDown() { VecSimIndex_Free(index); }

    virtual const void *GetDataByInternalId(idType id) = 0;

    template <typename algo_t>
    algo_t *CastIndex() {
        return dynamic_cast<algo_t *>(index);
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
