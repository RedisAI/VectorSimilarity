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

static int cmpVecSimQueryResultByScoreThenId(const VecSimQueryResult *res1,
                                             const VecSimQueryResult *res2) {
    return (VecSimQueryResult_GetScore(res1) != VecSimQueryResult_GetScore(res2))
               ? cmpVecSimQueryResultByScore(res1, res2)
               : cmpVecSimQueryResultById(res1, res2);
}

// Assumes that the arrays are sorted by score firstly and by id secondarily.
template <bool withSet>
VecSimQueryResult_List merge_results(VecSimQueryResult_List first, VecSimQueryResult_List second,
                                     size_t limit) {

    VecSimQueryResult *results = array_new<VecSimQueryResult>(limit);
    VecSimQueryResult_List mergedResults{.results = results, .code = VecSim_QueryResult_OK};
    VecSimQueryResult *cur_first = first.results;
    VecSimQueryResult *cur_second = second.results;
    const auto first_end = cur_first + VecSimQueryResult_Len(first);
    const auto second_end = cur_second + VecSimQueryResult_Len(second);

    // Will hold the ids of the results we've already added to the merged results.
    // Will be used only if withSet is true.
    std::unordered_set<size_t> ids;

    while (limit && cur_first != first_end && cur_second != second_end) {
        int cmp = cmpVecSimQueryResultByScoreThenId(cur_first, cur_second);
        if (cmp > 0) {
            if (!withSet || ids.insert(cur_second->id).second) {
                array_append(results, *cur_second);
                limit--;
            }
            cur_second++;
        } else if (cmp < 0) {
            if (!withSet || ids.insert(cur_first->id).second) {
                array_append(results, *cur_first);
                limit--;
            }
            cur_first++;
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

    // If we didn't exit the loop because of he limit, at least one of the arrays is empty.
    // We can try appending the rest of the other array.
    if (limit != 0) {
        auto [cur, end] = cur_first == first_end ? std::make_pair(cur_second, second_end)
                                                 : std::make_pair(cur_first, first_end);
        while (limit && cur != end) {
            if (!withSet || ids.find(VecSimQueryResult_GetId(cur)) == ids.end()) {
                array_append(results, *cur);
                limit--;
            }
            cur++;
        }
    }

    VecSimQueryResult_Free(first);
    VecSimQueryResult_Free(second);
    return mergedResults;
}
