#pragma once

#include <stdlib.h>
#include <VecSim/query_results.h>
#include <utility>

template <typename dist_t>
struct CompareByFirst {
    constexpr bool operator()(std::pair<dist_t, uint> const &a,
                              std::pair<dist_t, uint> const &b) const noexcept {
        return (a.first != b.first) ? a.first < b.first : a.second < b.second;
    }
};

void float_vector_normalize(float *x, size_t dim);

void sort_results_by_id(VecSimQueryResult_List results);

void sort_results_by_score(VecSimQueryResult_List results);
