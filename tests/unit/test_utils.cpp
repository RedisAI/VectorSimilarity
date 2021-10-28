#include "test_utils.h"
#include "gtest/gtest.h"

/*
 * helper function to run Top K search and iterate over the results. ResCB is a callback that takes
 * the id, score and index of a result, and performs test-specific logic for each.
 */
void runTopKSearchTest(VecSimIndex *index, const void *query, size_t k,
                       std::function<void(int, float, int)> ResCB, VecSimQueryParams *params,
                       VecSimQueryResult_Order order) {
    VecSimQueryResult_List res =
        VecSimIndex_TopKQuery(index, (const void *)query, k, params, order);
    ASSERT_EQ(VecSimQueryResult_Len(res), k);
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = (int)VecSimQueryResult_GetId(item);
        float score = VecSimQueryResult_GetScore(item);
        ResCB(id, score, res_ind++);
    }
    ASSERT_EQ(res_ind, k);
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);
}
