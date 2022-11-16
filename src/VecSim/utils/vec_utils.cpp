/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "vec_utils.h"
#include "VecSim/query_result_struct.h"
#include <cmath>
#include <cassert>
#include <cerrno>
#include <climits>
#include <float.h>

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void *, const void *);
#endif

const char *VecSimCommonStrings::ALGORITHM_STRING = "ALGORITHM";
const char *VecSimCommonStrings::FLAT_STRING = "FLAT";
const char *VecSimCommonStrings::HNSW_STRING = "HNSW";

const char *VecSimCommonStrings::TYPE_STRING = "TYPE";
const char *VecSimCommonStrings::FLOAT32_STRING = "FLOAT32";
const char *VecSimCommonStrings::FLOAT64_STRING = "FLOAT64";
const char *VecSimCommonStrings::INT32_STRING = "INT32";
const char *VecSimCommonStrings::INT64_STRING = "INT64";

const char *VecSimCommonStrings::METRIC_STRING = "METRIC";
const char *VecSimCommonStrings::COSINE_STRING = "COSINE";
const char *VecSimCommonStrings::IP_STRING = "IP";
const char *VecSimCommonStrings::L2_STRING = "L2";

const char *VecSimCommonStrings::DIMENSION_STRING = "DIMENSION";
const char *VecSimCommonStrings::INDEX_SIZE_STRING = "INDEX_SIZE";
const char *VecSimCommonStrings::INDEX_LABEL_COUNT_STRING = "INDEX_LABEL_COUNT";
const char *VecSimCommonStrings::IS_MULTI_STRING = "IS_MULTI_VALUE";
const char *VecSimCommonStrings::MEMORY_STRING = "MEMORY";

const char *VecSimCommonStrings::HNSW_EF_RUNTIME_STRING = "EF_RUNTIME";
const char *VecSimCommonStrings::HNSW_M_STRING = "M";
const char *VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING = "EF_CONSTRUCTION";
const char *VecSimCommonStrings::HNSW_EPSILON_STRING = "EPSILON";
const char *VecSimCommonStrings::HNSW_MAX_LEVEL = "MAX_LEVEL";
const char *VecSimCommonStrings::HNSW_ENTRYPOINT = "ENTRYPOINT";

const char *VecSimCommonStrings::BLOCK_SIZE_STRING = "BLOCK_SIZE";
const char *VecSimCommonStrings::SEARCH_MODE_STRING = "LAST_SEARCH_MODE";
const char *VecSimCommonStrings::HYBRID_POLICY_STRING = "HYBRID_POLICY";
const char *VecSimCommonStrings::BATCH_SIZE_STRING = "BATCH_SIZE";

int cmpVecSimQueryResultById(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return (int)(VecSimQueryResult_GetId(res1) - VecSimQueryResult_GetId(res2));
}

int cmpVecSimQueryResultByScore(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    assert(!std::isnan(VecSimQueryResult_GetScore(res1)) &&
           !std::isnan(VecSimQueryResult_GetScore(res2)));
    // Compare doubles
    return (VecSimQueryResult_GetScore(res1) - VecSimQueryResult_GetScore(res2)) >= 0.0 ? 1 : -1;
}

void sort_results_by_id(VecSimQueryResult_List rl) {
    qsort(rl.results, VecSimQueryResult_Len(rl), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResultById);
}

void sort_results_by_score(VecSimQueryResult_List rl) {
    qsort(rl.results, VecSimQueryResult_Len(rl), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResultByScore);
}

VecSimResolveCode validate_positive_integer_param(VecSimRawParam rawParam, long long *val) {
    char *ep; // For checking that strtoll used all rawParam.valLen chars.
    errno = 0;
    *val = strtoll(rawParam.value, &ep, 0);
    // Here we verify that val is positive and strtoll was successful.
    // The last test checks that the entire rawParam.value was used.
    // We catch here inputs like "3.14", "123text" and so on.
    if (*val <= 0 || *val == LLONG_MAX || errno != 0 || (rawParam.value + rawParam.valLen) != ep) {
        return VecSimParamResolverErr_BadValue;
    }
    return VecSimParamResolver_OK;
}

VecSimResolveCode validate_positive_double_param(VecSimRawParam rawParam, double *val) {
    char *ep; // For checking that strtold used all rawParam.valLen chars.
    errno = 0;
    *val = strtod(rawParam.value, &ep);
    // Here we verify that val is positive and strtod was successful.
    // The last test checks that the entire rawParam.value was used.
    // We catch here inputs like "-3.14", "123text" and so on.
    if (*val <= 0 || *val == DBL_MAX || errno != 0 || (rawParam.value + rawParam.valLen) != ep) {
        return VecSimParamResolverErr_BadValue;
    }
    return VecSimParamResolver_OK;
}

const char *VecSimAlgo_ToString(VecSimAlgo vecsimAlgo) {
    switch (vecsimAlgo) {
    case VecSimAlgo_BF:
        return VecSimCommonStrings::FLAT_STRING;
    case VecSimAlgo_HNSWLIB:
        return VecSimCommonStrings::HNSW_STRING;
    default:
        return NULL;
    }
}

const char *VecSimType_ToString(VecSimType vecsimType) {
    switch (vecsimType) {
    case VecSimType_FLOAT32:
        return VecSimCommonStrings::FLOAT32_STRING;
    case VecSimType_FLOAT64:
        return VecSimCommonStrings::FLOAT64_STRING;
    case VecSimType_INT32:
        return VecSimCommonStrings::INT32_STRING;
    case VecSimType_INT64:
        return VecSimCommonStrings::INT64_STRING;
    default:
        return NULL;
    }
}

const char *VecSimMetric_ToString(VecSimMetric vecsimMetric) {
    switch (vecsimMetric) {
    case VecSimMetric_Cosine:
        return "COSINE";
    case VecSimMetric_IP:
        return "IP";
    case VecSimMetric_L2:
        return "L2";
    default:
        return NULL;
    }
}

const char *VecSimSearchMode_ToString(VecSearchMode vecsimSearchMode) {
    switch (vecsimSearchMode) {
    case EMPTY_MODE:
        return "EMPTY_MODE";
    case STANDARD_KNN:
        return "STANDARD_KNN";
    case HYBRID_ADHOC_BF:
        return "HYBRID_ADHOC_BF";
    case HYBRID_BATCHES:
        return "HYBRID_BATCHES";
    case HYBRID_BATCHES_TO_ADHOC_BF:
        return "HYBRID_BATCHES_TO_ADHOC_BF";
    case RANGE_QUERY:
        return "RANGE_QUERY";
    default:
        return NULL;
    }
}

size_t VecSimType_sizeof(VecSimType type) {
    switch (type) {
    case VecSimType_FLOAT32:
        return sizeof(float);
    case VecSimType_FLOAT64:
        return sizeof(double);
    case VecSimType_INT32:
        return sizeof(int32_t);
    case VecSimType_INT64:
        return sizeof(int64_t);
    }
    return 0;
}
