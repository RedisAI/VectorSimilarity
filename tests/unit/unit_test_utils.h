/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <functional>
#include <cmath>
#include <exception>
#include <thread>

#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "mock_thread_pool.h"
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

using DataTypeSet = ::testing::Types<IndexType<VecSimType_FLOAT32, float>
#ifdef FP64_TESTS
                                     ,
                                     IndexType<VecSimType_FLOAT64, double>
#endif
                                     >;

// Define index type for tests that can be automatically generated for single and multi.
template <VecSimType type, bool IsMulti, typename DataType, typename DistType = DataType>
struct IndexTypeExtended {
    static VecSimType get_index_type() { return type; }
    static bool isMulti() { return IsMulti; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

using DataTypeSetExtended = ::testing::Types<IndexTypeExtended<VecSimType_FLOAT32, false, float>,
                                             IndexTypeExtended<VecSimType_FLOAT32, true, float>
#ifdef FP64_TESTS
                                             ,
                                             IndexTypeExtended<VecSimType_FLOAT64, false, double>,
                                             IndexTypeExtended<VecSimType_FLOAT64, true, double>
#endif
                                             >;

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

template <typename data_t>
void CompareVectors(const data_t *v1, const data_t *v2, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        ASSERT_EQ(v1[i], v2[i]) << "Vectors are not equal at index " << i;
    }
}

inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .algoParams = {.hnswParams = HNSWParams{hnsw_params}}};
    return params;
}

inline VecSimParams CreateParams(const BFParams &bf_params) {
    VecSimParams params{.algo = VecSimAlgo_BF, .algoParams = {.bfParams = BFParams{bf_params}}};
    return params;
}

inline VecSimParams CreateParams(const TieredIndexParams &tiered_params) {
    VecSimParams params{.algo = VecSimAlgo_TIERED,
                        .algoParams = {.tieredParams = TieredIndexParams{tiered_params}}};
    return params;
}

inline VecSimParams CreateParams(const SVSParams &svs_params) {
    VecSimParams params{.algo = VecSimAlgo_SVS, .algoParams = {.svsParams = svs_params}};
    return params;
}

namespace test_utils {
template <typename IndexParams>
inline VecSimIndex *CreateNewIndex(IndexParams &index_params, VecSimType type,
                                   bool is_multi = false) {
    index_params.type = type;
    index_params.multi = is_multi;
    VecSimParams params = CreateParams(index_params);
    return VecSimIndex_New(&params);
}

TieredIndexParams CreateTieredParams(VecSimParams &primary_params,
                                     tieredIndexMock &mock_thread_pool);
VecSimIndex *CreateNewTieredHNSWIndex(const HNSWParams &hnsw_params,
                                      tieredIndexMock &mock_thread_pool);

extern VecsimQueryType query_types[4];

} // namespace test_utils

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
VecSimQueryParams CreateQueryParams(const SVSRuntimeParams &RuntimeParams);

inline void ASSERT_TYPE_EQ(double arg1, double arg2) { ASSERT_DOUBLE_EQ(arg1, arg2); }

inline void ASSERT_TYPE_EQ(float arg1, float arg2) { ASSERT_FLOAT_EQ(arg1, arg2); }
void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k, size_t expected_res_num,
                       std::function<void(size_t, double, size_t)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryReply_Order order = BY_SCORE);
void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(size_t, double, size_t)> ResCB,
                       VecSimQueryParams *params = nullptr,
                       VecSimQueryReply_Order order = BY_SCORE);

void runBatchIteratorSearchTest(VecSimBatchIterator *batch_iterator, size_t n_res,
                                std::function<void(size_t, double, size_t)> ResCB,
                                VecSimQueryReply_Order order = BY_SCORE,
                                size_t expected_n_res = -1);

void compareCommonInfo(CommonInfo info1, CommonInfo info2);
void compareFlatInfo(bfInfoStruct info1, bfInfoStruct info2);
void compareHNSWInfo(hnswInfoStruct info1, hnswInfoStruct info2);

void compareFlatIndexInfoToIterator(VecSimIndexDebugInfo info, VecSimDebugInfoIterator *infoIter);

void compareHNSWIndexInfoToIterator(VecSimIndexDebugInfo info, VecSimDebugInfoIterator *infoIter);

void compareTieredHNSWIndexInfoToIterator(VecSimIndexDebugInfo info,
                                          VecSimIndexDebugInfo frontendIndexInfo,
                                          VecSimIndexDebugInfo backendIndexInfo,
                                          VecSimDebugInfoIterator *infoIterator);

void compareSVSIndexInfoToIterator(VecSimIndexDebugInfo info, VecSimDebugInfoIterator *infoIter);

void runRangeQueryTest(VecSimIndex *index, const void *query, double radius,
                       const std::function<void(size_t, double, size_t)> &ResCB,
                       size_t expected_res_num, VecSimQueryReply_Order order = BY_ID,
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
// TODO: Move all test_utils to this namespace
namespace test_utils {
size_t CalcVectorDataSize(VecSimIndex *index, VecSimType data_type);

template <typename data_t, typename dist_t>
TieredHNSWIndex<data_t, dist_t> *cast_to_tiered_index(VecSimIndex *index) {
    return dynamic_cast<TieredHNSWIndex<data_t, dist_t> *>(index);
}

} // namespace test_utils

// Test a specific exception type is thrown and prints the right message.
#define ASSERT_EXCEPTION_MESSAGE(VALUE, EXCEPTION_TYPE, MESSAGE)                                   \
    try {                                                                                          \
        VALUE;                                                                                     \
        FAIL() << "exception '" << MESSAGE << "' not thrown at all!";                              \
    } catch (const EXCEPTION_TYPE &e) {                                                            \
        EXPECT_EQ(std::string(MESSAGE), e.what())                                                  \
            << " exception message is incorrect. Expected the following "                          \
               "message:\n\n"                                                                      \
            << MESSAGE << "\n";                                                                    \
    } catch (...) {                                                                                \
        FAIL() << "exception '" << MESSAGE << "' not thrown with expected type '"                  \
               << #EXCEPTION_TYPE << "'!";                                                         \
    }
