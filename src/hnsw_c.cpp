#include "hnsw_c.h"
#include "../hnswlib/hnswlib/hnswalg.h"
#include "../hnswlib/hnswlib/bruteforce.h"

using namespace std;
using namespace hnswlib;

struct BFIndex {
    BruteforceSearch<float>* bf;
};
struct HNSWIndex {
    HierarchicalNSW<float>* hnsw;
};

#ifdef __cplusplus
extern "C" {
#endif

BFIndex *InitBFIndex() {
    int d = 4;
    size_t n = 100;
    auto space = new L2Space(d);  // We need to delete it in the end
    auto *bf = new BruteforceSearch<float>(space, 2 * n);
    return new BFIndex{bf};
}

HNSWIndex *InitHNSWIndex() {
    int d = 4;
    size_t n = 100;
    auto space = new L2Space(d); // We need to delete it in the end
    auto *hnsw = new HierarchicalNSW<float>(space, 2 * n);
    return new HNSWIndex{hnsw};
}

bool AddVectorToBFIndex(BFIndex *index, const void* vector_data, size_t id) {
    try {
        index->bf->addPoint(vector_data, id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

bool AddVectorToHNSWIndex(HNSWIndex *index, const void* vector_data, size_t id) {
    try {
        index->hnsw->addPoint(vector_data, id);
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

bool RemoveVectorFromHNSWIndex(HNSWIndex *index, size_t id) {
    try {
        index->hnsw->markDelete(id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

size_t GetBFIndexSize(BFIndex *index) {
    return index->bf->cur_element_count;
}

size_t GetHNSWIndexSize(HNSWIndex *index) {
    return index->hnsw->cur_element_count;
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

Vector *HNSWSearch(HNSWIndex *index, const void* query_data, size_t k) {
    priority_queue<pair<float, size_t>> knn_res = index->hnsw->searchKnn(query_data, k);
    auto *results = (Vector *)calloc(k ,sizeof(Vector));
    for (int i = k-1; i >= 0; --i) {
        results[i] = Vector {knn_res.top().second, knn_res.top().first};
        knn_res.pop();
    }
    return results;
}

void RemoveBFIndex(BFIndex *index) {

    delete index->bf;
    delete index;
}

void RemoveHNSWIndex(HNSWIndex *index) {
    delete index->hnsw;
    delete index;
}

#ifdef __cplusplus
}
#endif
