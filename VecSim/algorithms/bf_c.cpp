#include "bf_c.h"
#include "hnswlib/hnswalg.h"

using namespace std;
using namespace hnswlib;


struct BFIndex {
    void* bf;
    L2Space* space;
};

#ifdef __cplusplus
extern "C" {
#endif

BFIndex *InitBFIndex(size_t max_elements, int d) {

    auto space = new L2Space(d);  // We need to delete it in the end
    auto *bf = new BruteforceSearch<float>(space, max_elements);
    return new BFIndex{bf, space};
}

bool AddVectorToBFIndex(BFIndex *index, const void* vector_data, size_t id) {
    try {
        index->bf->addPoint(vector_data, id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

bool RemoveVectorFromBFIndex(BFIndex *index, size_t id) {
    try {
        index->bf->removePoint(id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

size_t GetBFIndexSize(BFIndex *index) {
    return index->bf->cur_element_count;
}

Vector *BFSearch(BFIndex *index, const void* query_data, size_t k) {
    priority_queue<pair<float, size_t>> knn_res = index->bf->searchKnn(query_data, k);
    auto *results = (Vector *)calloc(k ,sizeof(Vector));
    for (int i = k-1; i >= 0; --i) {
        results[i] = Vector {knn_res.top().second, knn_res.top().first};
        knn_res.pop();
    }
    return results;
}

void RemoveBFIndex(BFIndex *index) {

    delete index->bf;
    delete index->space;
    delete index;
}

#ifdef __cplusplus
}
#endif
