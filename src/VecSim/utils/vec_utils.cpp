#include "vec_utils.h"
#include "VecSim/query_result_struct.h"
#include <cmath>
#include <cassert>

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
const char *VecSimCommonStrings::MEMORY_STRING = "MEMORY";

const char *VecSimCommonStrings::HNSW_EF_RUNTIME_STRING = "EF_RUNTIME";
const char *VecSimCommonStrings::HNSW_M_STRING = "M";
const char *VecSimCommonStrings::HNSW_EF_CONSTRUCTION_STRING = "EF_CONSTRUCTION";
const char *VecSimCommonStrings::HNSW_MAX_LEVEL = "MAX_LEVEL";
const char *VecSimCommonStrings::HNSW_ENTRYPOINT = "ENTRYPOINT";

const char *VecSimCommonStrings::BLOCK_SIZE_STRING = "BLOCK_SIZE";
const char *VecSimCommonStrings::SEARCH_MODE_STRING = "LAST_SEARCH_MODE";

int cmpVecSimQueryResultById(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return (int)(VecSimQueryResult_GetId(res1) - VecSimQueryResult_GetId(res2));
}

int cmpVecSimQueryResultByScore(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    assert(!std::isnan(VecSimQueryResult_GetScore(res1)) &&
           !std::isnan(VecSimQueryResult_GetScore(res2)));
    // Compare floats
    return (VecSimQueryResult_GetScore(res1) - VecSimQueryResult_GetScore(res2)) >= 0.0 ? 1 : -1;
}

void float_vector_normalize(float *x, size_t dim) {
    double sum = 0;
    for (size_t i = 0; i < dim; i++) {
        sum += (double)x[i] * (double)x[i];
    }
    float norm = sqrt(sum);
    if (norm == 0)
        return;
    for (size_t i = 0; i < dim; i++) {
        x[i] = x[i] / norm;
    }
}

void sort_results_by_id(VecSimQueryResult_List results) {
    qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResultById);
}

void sort_results_by_score(VecSimQueryResult_List results) {
    qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
          (__compar_fn_t)cmpVecSimQueryResultByScore);
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
    default:
        return NULL;
    }
}
