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
    L2Space space(d);
    auto *bf = new BruteforceSearch<float>(&space, 2 * n);
    return new BFIndex{bf};
}

HNSWIndex *InitHNSWIndex() {
    int d = 4;
    size_t n = 100;
    L2Space space(d);
    auto *hnsw = new HierarchicalNSW<float>(&space, 2 * n);
    return new HNSWIndex{hnsw};
}

bool AddVectorToBFIndex(BFIndex *index, const char* vector_data, size_t id) {
    try {
        index->bf->addPoint(vector_data, id);
    } catch (exception &ex) {
        return false;
    }
    return true;
}

bool AddVectorToHNSWIndex(HNSWIndex *index, const char* vector_data, size_t id) {
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

#ifdef __cplusplus
}
#endif
