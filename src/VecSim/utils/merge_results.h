/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/query_result_struct.h"
#include <VecSim/utils/vec_utils.h>
#include "arr_cpp.h"
#include <unordered_set>

// Append the current result to the merged results, after verifying that it did not added yet (if
// verification is needed). Also update the set, limit and the current result.
template <bool withSet>
inline constexpr void maybe_append(VecSimQueryResult *&results, VecSimQueryResult *&cur_res,
                                   std::unordered_set<size_t> &ids, size_t &limit) {
    // In a single line, checks (only if a check is needed) if we already inserted the current id to
    // the merged results, add it to the set if not, and returns its conclusion.
    if (!withSet || ids.insert(cur_res->id).second) {
        array_append(results, *cur_res);
        limit--;
    }
    cur_res++;
}

// Assumes that the arrays are sorted by score firstly and by id secondarily.
// By the end of the function, the first and second referenced pointers will point to the first
// element that was not merged (in each array), or to the end of the array if it was merged
// completely.
template <bool withSet>
VecSimQueryResult *merge_results(VecSimQueryResult *&first, const VecSimQueryResult *first_end,
                                 VecSimQueryResult *&second, const VecSimQueryResult *second_end,
                                 size_t limit) {
    VecSimQueryResult *results = array_new<VecSimQueryResult>(limit);
    // Will hold the ids of the results we've already added to the merged results.
    // Will be used only if withSet is true.
    std::unordered_set<size_t> ids;
    auto &cur_first = first;
    auto &cur_second = second;

    while (limit && cur_first != first_end && cur_second != second_end) {
        int cmp = cmpVecSimQueryResultByScoreThenId(cur_first, cur_second);
        if (cmp > 0) {
            maybe_append<withSet>(results, cur_second, ids, limit);
        } else if (cmp < 0) {
            maybe_append<withSet>(results, cur_first, ids, limit);
        } else {
            // Even if `withSet` is true, we encountered an exact duplicate, so we know that this id
            // didn't appear before in both arrays, and it won't appear again in both arrays, so we
            // can add it to the merged results, and skip adding it to the set.
            array_append(results, *cur_first);
            cur_first++;
            cur_second++;
            limit--;
        }
    }

    // If we didn't exit the loop because of the limit, at least one of the arrays is empty.
    // We can try appending the rest of the other array.
    if (limit != 0) {
        if (cur_first == first_end) {
            while (limit && cur_second != second_end) {
                maybe_append<withSet>(results, cur_second, ids, limit);
            }
        } else {
            while (limit && cur_first != first_end) {
                maybe_append<withSet>(results, cur_first, ids, limit);
            }
        }
    }

    return results;
}

// Assumes that the arrays are sorted by score firstly and by id secondarily.
template <bool withSet>
VecSimQueryResult_List merge_result_lists(VecSimQueryResult_List first,
                                          VecSimQueryResult_List second, size_t limit) {

    VecSimQueryResult *cur_first = first.results;
    VecSimQueryResult *cur_second = second.results;
    const auto first_end = cur_first + VecSimQueryResult_Len(first);
    const auto second_end = cur_second + VecSimQueryResult_Len(second);

    auto results = merge_results<withSet>(cur_first, first_end, cur_second, second_end, limit);
    VecSimQueryResult_List mergedResults{.results = results, .code = VecSim_QueryResult_OK};

    VecSimQueryResult_Free(first);
    VecSimQueryResult_Free(second);
    return mergedResults;
}

// Concatenate the results of two queries into the results of the first query, consuming the second.
static inline void concat_results(VecSimQueryResult_List &first, VecSimQueryResult_List &second) {
    auto &dst = first.results;
    auto &src = second.results;

    dst = array_concat(dst, src);
    VecSimQueryResult_Free(second);
}
