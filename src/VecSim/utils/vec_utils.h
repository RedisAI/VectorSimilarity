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

    static const char *BLOCK_SIZE_STRING;
    static const char *SEARCH_MODE_STRING;
    static const char *HYBRID_POLICY_STRING;
    static const char *BATCH_SIZE_STRING;
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

// Sorts the results by id and removes duplicates.
// Assumes that a result can appear at most twice in the results list.
// @returns the number of unique results. This should be set to be the new length of the results
template <bool IsMulti>
size_t filter_results_by_id(VecSimQueryResult_List results) {
    if (VecSimQueryResult_Len(results) < 2) {
        return VecSimQueryResult_Len(results);
    }
    sort_results_by_id(results);

    size_t i, cur_end;
    for (i = 0, cur_end = 0; i < VecSimQueryResult_Len(results) - 1; i++, cur_end++) {
        const VecSimQueryResult *cur_res = results.results + i;
        const VecSimQueryResult *next_res = cur_res + 1;
        if (VecSimQueryResult_GetId(cur_res) == VecSimQueryResult_GetId(next_res)) {
            if (IsMulti) {
                // On multi value index, scores might be different and we want to keep the lower
                // score.
                if (VecSimQueryResult_GetScore(cur_res) < VecSimQueryResult_GetScore(next_res)) {
                    results.results[cur_end] = *cur_res;
                } else {
                    results.results[cur_end] = *next_res;
                }
            } else {
                // On single value index, scores are the same so we can keep any of the results.
                results.results[cur_end] = *cur_res;
            }
            // Assuming every id can appear at most twice, we can skip the next comparison between
            // the current and the next result.
            i++;
        } else {
            results.results[cur_end] = *cur_res;
        }
    }
    return cur_end + 1;
}
