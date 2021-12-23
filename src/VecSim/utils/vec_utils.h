#pragma once

#include <stdlib.h>
#include "VecSim/vec_sim_common.h"
#include <VecSim/query_results.h>
#include <utility>

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
    static const char *MEMORY_STRING;

    static const char *HNSW_EF_RUNTIME_STRING;
    static const char *HNSW_EF_CONSTRUCTION_STRING;
    static const char *HNSW_M_STRING;
    static const char *HNSW_MAX_LEVEL;
    static const char *HNSW_ENTRYPOINT;

    static const char *BLOCK_SIZE_STRING;
};

void float_vector_normalize(float *x, size_t dim);

void sort_results_by_id(VecSimQueryResult_List results);

void sort_results_by_score(VecSimQueryResult_List results);

const char *VecSimAlgo_ToString(VecSimAlgo vecsimAlgo);

const char *VecSimType_ToString(VecSimType vecsimType);

const char *VecSimMetric_ToString(VecSimMetric vecsimMetric);
