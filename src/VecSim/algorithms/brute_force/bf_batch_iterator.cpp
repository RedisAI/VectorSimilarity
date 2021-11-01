#include "bf_batch_iterator.h"

BF_BatchIterator::BF_BatchIterator(const void *query_vector, const VecSimIndex *index) :
    VecSimBatchIterator(query_vector, index) {
        id = 0;
        lower_bound = std::numeric_limits<float>::max();
}

VecSimQueryResult_List BF_BatchIterator::getNextResults(size_t n_res) {

}






