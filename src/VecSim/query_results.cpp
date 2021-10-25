#include <limits>
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"

struct VecSimQueryResult_Iterator {
    VecSimQueryResult *curr_result;
    size_t index;
    size_t results_len;
};

struct VecSimBatchIterator {
    VecSimIndex *index;
    unsigned char iterator_id;
    size_t returned_results_count;

    BatchIterator_Next IteratorNext;
    BatchIterator_HasNext IteratorHasNext;
    BatchIterator_Free IteratorFree;
    BatchIterator_Reset IteratorReset;
};

extern "C" size_t VecSimQueryResult_Len(VecSimQueryResult_List *result) {
    return array_len((VecSimQueryResult *)result);
}

extern "C" void VecSimQueryResult_Free(VecSimQueryResult_List *result) {
    array_free((VecSimQueryResult *)result);
}

extern "C" VecSimQueryResult_Iterator *
VecSimQueryResult_List_GetIterator(VecSimQueryResult_List *results) {
    return new VecSimQueryResult_Iterator{(VecSimQueryResult *)results, 0,
                                          VecSimQueryResult_Len(results)};
}

extern "C" bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->index == iterator->results_len) {
        return false;
    }
    return true;
}

extern "C" VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->index == iterator->results_len) {
        return nullptr;
    }
    VecSimQueryResult *item = iterator->curr_result++;
    iterator->index++;

    return item;
}

extern "C" int VecSimQueryResult_GetId(VecSimQueryResult *res) {
    if (res == nullptr) {
        return -1;
    }
    return (int)res->id;
}

extern "C" float VecSimQueryResult_GetScore(VecSimQueryResult *res) {
    if (res == nullptr) {
        return std::numeric_limits<float>::min(); // "minus infinity"
    }
    return (float)res->score;
}

extern "C" void VecSimQueryResult_IteratorFree(VecSimQueryResult_Iterator *iterator) {
    delete iterator;
}

VecSimQueryResult VecSimQueryResult_Create(size_t id, float score) {
    return VecSimQueryResult{id, score};
}

void VecSimQueryResult_SetId(VecSimQueryResult result, size_t id) { result.id = id; }

void VecSimQueryResult_SetScore(VecSimQueryResult result, float score) { result.score = score; }
