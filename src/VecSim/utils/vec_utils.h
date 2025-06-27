/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <stdlib.h>
#include "VecSim/vec_sim_common.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/vecsim_stl.h"
#include <utility>
#include <cassert>

struct VecSimCommonStrings {
public:
    static const char *ALGORITHM_STRING;
    static const char *FLAT_STRING;
    static const char *HNSW_STRING;
    static const char *TIERED_STRING;
    static const char *SVS_STRING;

    static const char *TYPE_STRING;
    static const char *FLOAT32_STRING;
    static const char *FLOAT64_STRING;
    static const char *BFLOAT16_STRING;
    static const char *FLOAT16_STRING;
    static const char *INT8_STRING;
    static const char *UINT8_STRING;
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

    static const char *SVS_WS_SEARCH_STRING;
    static const char *SVS_BC_SEARCH_STRING;
    static const char *SVS_USE_SEARCH_HISTORY_STRING;

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

VecSimResolveCode validate_vecsim_bool_param(VecSimRawParam rawParam, VecSimOptionMode *val);

const char *VecSimAlgo_ToString(VecSimAlgo vecsimAlgo);

const char *VecSimType_ToString(VecSimType vecsimType);

const char *VecSimMetric_ToString(VecSimMetric vecsimMetric);

const char *VecSimSearchMode_ToString(VecSearchMode vecsimSearchMode);

size_t VecSimType_sizeof(VecSimType vecsimType);

/** Returns the size in bytes of a stored or query vector */
size_t VecSimParams_GetDataSize(VecSimType type, size_t dim, VecSimMetric metric);
