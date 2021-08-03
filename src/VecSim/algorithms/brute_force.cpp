#include "brute_force.h"
#include <unordered_map>
#include <set>
#include <vector>

using namespace std;

typedef size_t labelType;
typedef size_t idType;

// Predeclaration

struct VectorBlock;

struct VectorBlockMember{
    size_t index;
    VectorBlock* block;
    labelType label;
};

struct VectorBlock {
    size_t size;
    VectorBlockMember* members;
    float vectors[];
};

struct BruteForceIndex {
    BruteForceIndex(VecSimType vectype, VecSimMetric metric, size_t dim, size_t max_elements);

    // Meta data
    VecSimIndex base;
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
    // TODO: support cosine, angular
    

    idType count;
    std::unordered_map<labelType, idType> labelToIdLookup;
    std::set<idType> deletedIds;
    std::vector<VectorBlock*> vectorBlocks;
};

extern "C" VecSimIndex *BruteForce_New(VecSimParams *params) {

}

extern "C" void BruteForce_Free(VecSimIndex *index) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    for (auto& vectorBlock : bfIndex->vectorBlocks) {
        delete vectorBlock->vectors;
        delete[] vectorBlock->vectors;
    }
}

static void BruteForce_UpdateVector(BruteForceIndex* bfIndex, idType id, const void *vector_data) {

}

extern "C" int BruteForce_AddVector(VecSimIndex *index, const void *vector_data, size_t label) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);

    idType id;
    bool update = false;
    auto optionalID = bfIndex->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != bfIndex->labelToIdLookup.end()) {
        id = optionalID->second;
        BruteForce_UpdateVector(bfIndex, id, vector_data);
        return true;
    }
    else {
        // Try re-use deleted id.
        if(bfIndex->deletedIds.size()!=0) {
            id = *bfIndex->deletedIds.begin();
            bfIndex->deletedIds.erase(bfIndex->deletedIds.begin());
        }
        else {
            id = bfIndex->count++;
        }
    }

    
    
}

extern "C" int BruteForce_DeleteVector(VecSimIndex *index, size_t label) {

}

extern "C" size_t BruteForce_Size(VecSimIndex *index);

extern "C" VecSimQueryResult *BruteForce_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                  VecSimQueryParams *queryParams);

extern "C" VecSimIndexInfo BruteForce_Info(VecSimIndex *index);

// TODO

extern "C" VecSimQueryResult *BruteForce_DistanceQuery(VecSimIndex *index, const void *queryBlob, float distance,
                                      VecSimQueryParams queryParams);

extern "C" void BruteForce_ClearDeleted(VecSimIndex *index);


BruteForceIndex::BruteForceIndex(VecSimType vectype, VecSimMetric metric, size_t dim, size_t max_elements):
metric(metric), vecType(vecType), dim(dim) {
    base = VecSimIndex{
        AddFn : BruteForce_AddVector,
        DeleteFn: BruteForce_DeleteVector,
        SizeFn: BruteForce_Size,
        TopKQueryFn: BruteForce_TopKQuery,
        DistanceQueryFn : NULL,
        ClearDeletedFn : NULL,
        FreeFn : BruteForce_Free,
        InfoFn : BruteForce_Info
    };
}
