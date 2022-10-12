#pragma once

#include <functional>
#include <cmath>
#include <exception>

#include "VecSim/vec_sim.h"
#include "gtest/gtest.h"

// IndexType is used to define indices unit tests
template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

#define TEST_DATA_T typename TypeParam::data_t
#define TEST_DIST_T typename TypeParam::dist_t

using DataTypeSet =
    ::testing::Types<IndexType<VecSimType_FLOAT32, float>, IndexType<VecSimType_FLOAT64, double>>;

template <typename params_t>
class IndexTestUtils {

protected:
    IndexTestUtils(VecSimType type, bool is_multi) : params{} {
        params.type = type;
        params.multi = is_multi;
    }
    params_t params;
};

template <typename data_t>
static void GenerateVector(data_t *output, size_t dim, data_t value = 1.0) {
    for (size_t i = 0; i < dim; i++) {
        output[i] = (data_t)value;
    }
}

template <typename data_t>
int GenerateAndAddVector(VecSimIndex *index, size_t dim, size_t id, data_t value = 1.0) {
    data_t v[dim];
    GenerateVector(v, dim, value);
    return VecSimIndex_AddVector(index, v, id);
}

inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
    return params;
}

inline VecSimParams CreateParams(const BFParams &bf_params) {
    VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
    return params;
}
template <typename IndexParams>
VecSimIndex *CreateNewIndex(const IndexParams &index_params) {
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_New(&params);
}

template <typename IndexParams>
size_t EstimateInitialSize(const IndexParams &index_params) {
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_EstimateInitialSize(&params);
}

template <typename IndexParams>
size_t EstimateElementSize(const IndexParams &index_params) {
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_EstimateElementSize(&params);
}

VecSimQueryParams CreateQueryParams(const HNSWRuntimeParams &RuntimeParams);

inline void ASSERT_TYPE_EQ(double arg1, double arg2) { ASSERT_DOUBLE_EQ(arg1, arg2); }

inline void ASSERT_TYPE_EQ(float arg1, float arg2) { ASSERT_FLOAT_EQ(arg1, arg2); }
void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryResult_Order order = BY_SCORE);

void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(size_t, double, size_t)> ResCB,
                                VecSimQueryResult_Order order = BY_SCORE,
                                size_t expected_n_res = -1);

void compareFlatIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void compareHNSWIndexInfoToIterator(VecSimIndexInfo info, VecSimInfoIterator *infoIter);

void runRangeQueryTest(VecSimIndex *index, const void *query, double radius,
                       const std::function<void(size_t, double, size_t)> &ResCB,
                       size_t expected_res_num, VecSimQueryResult_Order order = BY_ID,
                       VecSimQueryParams *params = nullptr);

size_t getLabelsLookupNodeSize();

inline double GetInfVal(VecSimType type) {
    if (type == VecSimType_FLOAT64) {
        return exp(500);
    } else if (type == VecSimType_FLOAT32) {
        return exp(60);
    } else {
        throw std::invalid_argument("This type is not supported");
    }
}
