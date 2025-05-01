/*
* Copyright (c) 2006-Present, Redis Ltd.
* All rights reserved.
*
* Licensed under your choice of the Redis Source Available License 2.0
* (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
* GNU Affero General Public License v3 (AGPLv3).
*/


#include "bm_vecsim_general.h"

void BM_VecSimGeneral::MeasureRecall(VecSimQueryReply *hnsw_results, VecSimQueryReply *bf_results,
                                     std::atomic_int &correct) {
    auto hnsw_it = VecSimQueryReply_GetIterator(hnsw_results);
    while (VecSimQueryReply_IteratorHasNext(hnsw_it)) {
        auto hnsw_res_item = VecSimQueryReply_IteratorNext(hnsw_it);
        auto bf_it = VecSimQueryReply_GetIterator(bf_results);
        while (VecSimQueryReply_IteratorHasNext(bf_it)) {
            auto bf_res_item = VecSimQueryReply_IteratorNext(bf_it);
            if (VecSimQueryResult_GetId(hnsw_res_item) == VecSimQueryResult_GetId(bf_res_item)) {
                correct++;
                break;
            }
        }
        VecSimQueryReply_IteratorFree(bf_it);
    }
    VecSimQueryReply_IteratorFree(hnsw_it);
}
