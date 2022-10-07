#include "VecSim/query_result_struct.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/vec_sim.h"
#include "VecSim/batch_iterator.h"

struct VecSimQueryResult_Iterator {
    VecSimQueryResult *first_result;
    VecSimQueryResult *curr_result;
    size_t index;
    size_t results_len;

    explicit VecSimQueryResult_Iterator(VecSimQueryResult_List results_array)
        : first_result(results_array.results), curr_result(results_array.results), index(0),
          results_len(array_len(results_array.results)) {}
};

extern "C" size_t VecSimQueryResult_Len(VecSimQueryResult_List rl) { return array_len(rl.results); }

extern "C" size_t VecSimQueryResult_ArrayLen(VecSimQueryResult *rl) { return array_len(rl); }

extern "C" void VecSimQueryResult_Free(VecSimQueryResult_List rl) {
    if (rl.results) {
        array_free(rl.results);
        rl.results = NULL;
    }
}

extern "C" void VecSimQueryResult_FreeArray(VecSimQueryResult *rl) {
    if (rl) {
        array_free(rl);
    }
}

extern "C" VecSimQueryResult_Iterator *
VecSimQueryResult_List_GetIterator(VecSimQueryResult_List results) {
    return new VecSimQueryResult_Iterator(results);
}

extern "C" bool VecSimQueryResult_IteratorHasNext(VecSimQueryResult_Iterator *iterator) {
    return iterator->index != iterator->results_len;
}

extern "C" VecSimQueryResult *VecSimQueryResult_IteratorNext(VecSimQueryResult_Iterator *iterator) {
    if (iterator->index == iterator->results_len) {
        return nullptr;
    }
    VecSimQueryResult *item = iterator->curr_result++;
    iterator->index++;

    return item;
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
    iterator->index = 0;
    iterator->curr_result = iterator->first_result;
}

/********************** batch iterator API ***************************/
VecSimQueryResult_List VecSimBatchIterator_Next(VecSimBatchIterator *iterator, size_t n_results,
                                                VecSimQueryResult_Order order) {
    return iterator->getNextResults(n_results, order);
}

bool VecSimBatchIterator_HasNext(VecSimBatchIterator *iterator) { return !iterator->isDepleted(); }

void VecSimBatchIterator_Free(VecSimBatchIterator *iterator) {
    // Batch iterator might be deleted after the index, so it should keep the allocator before
    // deleteing.
    auto allocator = iterator->getAllocator();
    delete iterator;
}

void VecSimBatchIterator_Reset(VecSimBatchIterator *iterator) { iterator->reset(); }
