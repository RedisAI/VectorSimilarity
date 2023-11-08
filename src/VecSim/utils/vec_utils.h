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

struct VecSimCommonStrings {
public:
    static const char *ALGORITHM_STRING;
    static const char *FLAT_STRING;
    static const char *HNSW_STRING;
    static const char *RAFTIVFFLAT_STRING;
    static const char *RAFTIVFPQ_STRING;
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

    // Log levels
    static const char *LOG_DEBUG_STRING;
    static const char *LOG_VERBOSE_STRING;
    static const char *LOG_NOTICE_STRING;
    static const char *LOG_WARNING_STRING;
};

void sort_results_by_id(VecSimQueryReply *results);

void sort_results_by_score(VecSimQueryReply *results);

void sort_results_by_score_then_id(VecSimQueryReply *results);

void sort_results(VecSimQueryReply *results, VecSimQueryReply_Order order);

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
