/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/
#pragma once

#include "VecSim/query_result_definitions.h"
#include <VecSim/utils/vec_utils.h>

// Compare two results by score, and if the scores are equal, by id.
inline int cmpVecSimQueryResultByScoreThenId(const VecSimQueryResultContainer::iterator res1,
                                             const VecSimQueryResultContainer::iterator res2) {
    return (res1->score != res2->score) ? (res1->score > res2->score ? 1 : -1)
                                        : (int)(res1->id - res2->id);
}

// Append the current result to the merged results, after verifying that it did not added yet (if
// verification is needed). Also update the set, limit and the current result.
template <bool withSet>
inline constexpr void maybe_append(VecSimQueryResultContainer &results,
                                   VecSimQueryResultContainer::iterator &cur_res,
                                   std::unordered_set<size_t> &ids, size_t &limit) {
    // In a single line, checks (only if a check is needed) if we already inserted the current id to
    // the merged results, add it to the set if not, and returns its conclusion.
    if (!withSet || ids.insert(cur_res->id).second) {
        results.push_back(*cur_res);
        limit--;
    }
    cur_res++;
}

// Assumes that the arrays are sorted by score firstly and by id secondarily.
// By the end of the function, the first and second referenced pointers will point to the first
// element that was not merged (in each array), or to the end of the array if it was merged
// completely.
template <bool withSet>
std::pair<size_t, size_t> merge_results(VecSimQueryResultContainer &results,
                                        VecSimQueryResultContainer &first,
                                        VecSimQueryResultContainer &second, size_t limit) {
    // Allocate the merged results array with the minimum size needed.
    // Min of the limit and the sum of the lengths of the two arrays.
    results.reserve(std::min(limit, first.size() + second.size()));
    // Will hold the ids of the results we've already added to the merged results.
    // Will be used only if withSet is true.
    std::unordered_set<size_t> ids;
    auto cur_first = first.begin();
    auto cur_second = second.begin();

    while (limit && cur_first != first.end() && cur_second != second.end()) {
        int cmp = cmpVecSimQueryResultByScoreThenId(cur_first, cur_second);
        if (cmp > 0) {
            maybe_append<withSet>(results, cur_second, ids, limit);
        } else if (cmp < 0) {
            maybe_append<withSet>(results, cur_first, ids, limit);
        } else {
            // Even if `withSet` is true, we encountered an exact duplicate, so we know that this id
            // didn't appear before in both arrays, and it won't appear again in both arrays, so we
            // can add it to the merged results, and skip adding it to the set.
            results.push_back(*cur_first);
            cur_first++;
            cur_second++;
            limit--;
        }
    }

    // If we didn't exit the loop because of the limit, at least one of the arrays is empty.
    // We can try appending the rest of the other array.
    if (limit != 0) {
        if (cur_first == first.end()) {
            while (limit && cur_second != second.end()) {
                maybe_append<withSet>(results, cur_second, ids, limit);
            }
        } else {
            while (limit && cur_first != first.end()) {
                maybe_append<withSet>(results, cur_first, ids, limit);
            }
        }
    }

    // Return the number of elements that were merged from each array.
    return {cur_first - first.begin(), cur_second - second.begin()};
}

// Assumes that the arrays are sorted by score firstly and by id secondarily.
template <bool withSet>
VecSimQueryReply *merge_result_lists(VecSimQueryReply *first, VecSimQueryReply *second,
                                     size_t limit) {

    auto mergedResults = new VecSimQueryReply(first->results.getAllocator());
    merge_results<withSet>(mergedResults->results, first->results, second->results, limit);

    VecSimQueryReply_Free(first);
    VecSimQueryReply_Free(second);
    return mergedResults;
}

// Concatenate the results of two queries into the results of the first query, consuming the second.
static inline void concat_results(VecSimQueryReply *first, VecSimQueryReply *second) {
    first->results.insert(first->results.end(), second->results.begin(), second->results.end());
    VecSimQueryReply_Free(second);
}

// Sorts the results by id and removes duplicates.
// Assumes that a result can appear at most twice in the results list.
// @returns the number of unique results. This should be set to be the new length of the results
template <bool IsMulti>
void filter_results_by_id(VecSimQueryReply *results) {
    if (VecSimQueryReply_Len(results) < 2) {
        return;
    }
    sort_results_by_id(results);

    size_t i, cur_end;
    for (i = 0, cur_end = 0; i < VecSimQueryReply_Len(results) - 1; i++, cur_end++) {
        const VecSimQueryResult *cur_res = results->results.data() + i;
        const VecSimQueryResult *next_res = cur_res + 1;
        if (VecSimQueryResult_GetId(cur_res) == VecSimQueryResult_GetId(next_res)) {
            if (IsMulti) {
                // On multi value index, scores might be different and we want to keep the lower
                // score.
                if (VecSimQueryResult_GetScore(cur_res) < VecSimQueryResult_GetScore(next_res)) {
                    results->results[cur_end] = *cur_res;
                } else {
                    results->results[cur_end] = *next_res;
                }
            } else {
                // On single value index, scores are the same so we can keep any of the results.
                results->results[cur_end] = *cur_res;
            }
            // Assuming every id can appear at most twice, we can skip the next comparison between
            // the current and the next result.
            i++;
        } else {
            results->results[cur_end] = *cur_res;
        }
    }
    // If the last result is unique, we need to add it to the results.
    if (i == VecSimQueryReply_Len(results) - 1) {
        results->results[cur_end++] = results->results[i];
        // Logically, we should increment cur_end and i here, but we don't need to because it won't
        // affect the rest of the function.
    }
    results->results.resize(cur_end);
}
