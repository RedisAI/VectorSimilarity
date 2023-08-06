/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/query_result_definitions.h"
#include "VecSim/vec_sim.h"
#include "VecSim/batch_iterator.h"
#include <assert.h>

struct VecSimQueryResult_Iterator {
    using iterator = vecsim_stl::vector<VecSimQueryResult>::iterator;
    const iterator begin, end;
    iterator current;

    explicit VecSimQueryResult_Iterator(VecSimQueryResult_List *result_list)
        : begin(result_list->results.begin()), end(result_list->results.end()), current(begin) {}
};

extern "C" size_t VecSimQueryResult_Len(VecSimQueryResult_List *rl) { return rl->results.size(); }

extern "C" VecSimQueryResult_Code VecSimQueryResult_GetCode(VecSimQueryResult_List *rl) {
    return rl->code;
}

extern "C" void VecSimQueryResult_Free(VecSimQueryResult_List *rl) {
    if (rl) {
        delete rl;
    }
}

extern "C" VecSimQueryResult_Iterator *
VecSimQueryResult_List_GetIterator(VecSimQueryResult_List *results) {
    return new VecSimQueryResult_Iterator(results);
}

extern "C" bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator) {
    return iterator->current != iterator->end;
}

extern "C" VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->current == iterator->end) {
        return nullptr;
    }

    return std::to_address(iterator->current++);
}

extern "C" int64_t VecSimQueryResult_GetId(const VecSimQueryResult *res) {
    if (res == nullptr) {
        return INVALID_ID;
    }
    return (int64_t)res->id;
}

extern "C" double VecSimQueryResult_GetScore(const VecSimQueryResult *res) {
    if (res == nullptr) {
        return INVALID_SCORE; // "NaN"
    }
    return res->score;
}

extern "C" void VecSimQueryResult_IteratorFree(VecSimQueryResult_Iterator *iterator) {
    delete iterator;
}

extern "C" void VecSimQueryResult_IteratorReset(VecSimQueryResult_Iterator *iterator) {
    iterator->current = iterator->begin;
}

/********************** batch iterator API ***************************/
VecSimQueryResult_List *VecSimBatchIterator_Next(VecSimBatchIterator *iterator, size_t n_results,
                                                 VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    return iterator->getNextResults(n_results, order);
}

bool VecSimBatchIterator_HasNext(VecSimBatchIterator *iterator) { return !iterator->isDepleted(); }

void VecSimBatchIterator_Free(VecSimBatchIterator *iterator) {
    // Batch iterator might be deleted after the index, so it should keep the allocator before
    // deleting.
    auto allocator = iterator->getAllocator();
    delete iterator;
}

void VecSimBatchIterator_Reset(VecSimBatchIterator *iterator) { iterator->reset(); }
