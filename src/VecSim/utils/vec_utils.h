/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include "VecSim/vec_sim_common.h"
#include <VecSim/query_results.h>
#include <utility>
#include <cassert>
#include <cmath> //sqrt

template <typename dist_t>
struct CompareByFirst {
    constexpr bool operator()(std::pair<dist_t, unsigned int> const &a,
                              std::pair<dist_t, unsigned int> const &b) const noexcept {
        return (a.first != b.first) ? a.first < b.first : a.second < b.second;
    }
};

struct VecSimCommonStrings {
public:
    static const char *ALGORITHM_STRING;
    static const char *FLAT_STRING;
    static const char *HNSW_STRING;
    static const char *TIERED_STRING;

    static const char *TYPE_STRING;
    static const char *FLOAT32_STRING;
    static const char *FLOAT64_STRING;
    static const char *INT32_STRING;
    static const char *INT64_STRING;

    static const char *METRIC_STRING;
    static const char *COSINE_STRING;
    static const char *IP_STRING;
    static const char *L2_STRING;

    static const char *DIMENSION_STRING;
    static const char *INDEX_SIZE_STRING;
    static const char *INDEX_LABEL_COUNT_STRING;
    static const char *IS_MULTI_STRING;
    static const char *MEMORY_STRING;

    static const char *HNSW_EF_RUNTIME_STRING;
    static const char *HNSW_EF_CONSTRUCTION_STRING;
    static const char *HNSW_M_STRING;
    static const char *HNSW_EPSILON_STRING;
    static const char *HNSW_MAX_LEVEL;
    static const char *HNSW_ENTRYPOINT;
    static const char *HNSW_NUM_MARKED_DELETED;
    // static const char *HNSW_VISITED_NODES_POOL_SIZE_STRING;

    static const char *BLOCK_SIZE_STRING;
    static const char *SEARCH_MODE_STRING;
    static const char *HYBRID_POLICY_STRING;
    static const char *BATCH_SIZE_STRING;

    static const char *TIERED_MANAGEMENT_MEMORY_STRING;
    static const char *TIERED_BACKGROUND_INDEXING_STRING;
    static const char *TIERED_BUFFER_LIMIT_STRING;
    static const char *FRONTEND_INDEX_STRING;
    static const char *BACKEND_INDEX_STRING;
    static const char *TIERED_HNSW_SWAP_JOBS_THRESHOLD_STRING;
};

inline int cmpVecSimQueryResultById(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return (int)(VecSimQueryResult_GetId(res1) - VecSimQueryResult_GetId(res2));
}

inline int cmpVecSimQueryResultByScore(const VecSimQueryResult *res1,
                                       const VecSimQueryResult *res2) {
    assert(!std::isnan(VecSimQueryResult_GetScore(res1)) &&
           !std::isnan(VecSimQueryResult_GetScore(res2)));
    // Compare doubles
    return (VecSimQueryResult_GetScore(res1) - VecSimQueryResult_GetScore(res2)) >= 0.0 ? 1 : -1;
}

inline int cmpVecSimQueryResultByScoreThenId(const VecSimQueryResult *res1,
                                             const VecSimQueryResult *res2) {
    return (VecSimQueryResult_GetScore(res1) != VecSimQueryResult_GetScore(res2))
               ? cmpVecSimQueryResultByScore(res1, res2)
               : cmpVecSimQueryResultById(res1, res2);
}

void sort_results_by_id(VecSimQueryResult_List results);

void sort_results_by_score(VecSimQueryResult_List results);

void sort_results_by_score_then_id(VecSimQueryResult_List results);

void sort_results(VecSimQueryResult_List results, VecSimQueryResult_Order order);

VecSimResolveCode validate_positive_integer_param(VecSimRawParam rawParam, long long *val);

VecSimResolveCode validate_positive_double_param(VecSimRawParam rawParam, double *val);

const char *VecSimAlgo_ToString(VecSimAlgo vecsimAlgo);

const char *VecSimType_ToString(VecSimType vecsimType);

const char *VecSimMetric_ToString(VecSimMetric vecsimMetric);

const char *VecSimSearchMode_ToString(VecSearchMode vecsimSearchMode);

size_t VecSimType_sizeof(VecSimType vecsimType);

template <typename DataType>
void normalizeVector(DataType *input_vector, size_t dim) {

    // Cast to double to avoid float overflow.
    double sum = 0;

    for (size_t i = 0; i < dim; i++) {
        sum += (double)input_vector[i] * (double)input_vector[i];
    }
    DataType norm = sqrt(sum);

    for (size_t i = 0; i < dim; i++) {
        input_vector[i] = input_vector[i] / norm;
    }
}

typedef void (*normalizeVector_f)(void *input_vector, size_t dim);

static inline void normalizeVectorFloat(void *input_vector, size_t dim) {
    normalizeVector(static_cast<float *>(input_vector), dim);
}
static inline void normalizeVectorDouble(void *input_vector, size_t dim) {
    normalizeVector(static_cast<double *>(input_vector), dim);
}
