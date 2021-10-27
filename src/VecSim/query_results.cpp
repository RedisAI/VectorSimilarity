#include "VecSim/query_results.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/vec_sim.h"

struct VecSimQueryResult_Iterator {
    VecSimQueryResult *curr_result;
    size_t index;
    size_t results_len;

    explicit VecSimQueryResult_Iterator(VecSimQueryResult *results_array)
        : curr_result(results_array), index(0), results_len(array_len(results_array)) {}
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
    return new VecSimQueryResult_Iterator((VecSimQueryResult *)results);
}

extern "C" bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->index == iterator->results_len) {
        return false;
    }
    return true;
}

extern "C" VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator) {
    if (!VecSimQueryResult_IteratorHasNext(iterator)) {
        return nullptr;
    }
    VecSimQueryResult *item = iterator->curr_result++;
    iterator->index++;

    return item;
}

extern "C" long VecSimQueryResult_GetId(VecSimQueryResult *res) {
    if (res == nullptr) {
        return INVALID_ID;
    }
    return (long)res->id;
}

extern "C" float VecSimQueryResult_GetScore(VecSimQueryResult *res) {
    if (res == nullptr) {
        return INVALID_SCORE; // "minus infinity"
    }
    return (float)res->score;
}

extern "C" void VecSimQueryResult_IteratorFree(VecSimQueryResult_Iterator *iterator) {
    delete iterator;
}
