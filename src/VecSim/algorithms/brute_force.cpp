#include "brute_force.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <cstring>
#include <queue>
#include "cblas.h"
#include "lapack.h"
#include <limits>
#include "VecSim/utils/arr_cpp.h"

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
    VectorBlockMember** members;
    float vectors[];
};

typedef void (*DistanceCalculateFunction)(size_t dim, VectorBlock* vectorBlock, const void* queryBlob, float* scores);

void BruteForceIndex_InternalProduct(size_t dim, VectorBlock* vectorBlock, const void* queryBlob, float* scores){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, vectorBlock->size, dim, 1, vectorBlock->vectors,  dim, (const float *)queryBlob, 1 , 0 , scores, 1);
}

void BruteForceIndex_L2(size_t dim, VectorBlock* vectorBlock, const void* queryBlob, float* scores){
    float tmp_vector[dim];
    for(size_t i = 0; i < vectorBlock->size; i++) {
        cblas_scopy(dim, vectorBlock->vectors+(i*dim), 1, tmp_vector, 1);
        cblas_saxpy(dim, -1.0f, (const float*) queryBlob, 1, tmp_vector, 1 );
        scores[i] = cblas_sdot(dim, tmp_vector, 1, tmp_vector, 1);
    }
}

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
    std::vector<VectorBlockMember*> idToVectorBlockMemberMapping;
    std::set<idType> deletedIds;
    std::vector<VectorBlock*> vectorBlocks;
    size_t vectorBlockSize;

    DistanceCalculateFunction distanceCalculationFunction;
};

struct CompareByFirst {
    constexpr bool operator()(std::pair<float, labelType> const &a,
                                std::pair<float, labelType> const &b) const noexcept {
        return a.first < b.first;
    }
};

static VectorBlock* VectorBlock_new(size_t vectorCount, size_t vectorDim) {
    VectorBlock* vectorBlock = (VectorBlock*)calloc(1, sizeof(VectorBlock) + vectorCount*vectorDim);
    vectorBlock->members = (VectorBlockMember**)calloc(vectorCount, sizeof(VectorBlockMember*));
    return vectorBlock;
}

static void VectorBlock_Delete(VectorBlock* vectorBlock) {
    if(!vectorBlock) {
        return;
    }

    delete vectorBlock->members;
    delete vectorBlock;
}

extern "C" VecSimIndex *BruteForce_New(VecSimParams *params) {
        try {
        auto p = new BruteForceIndex(params->type, params->metric, params->size, params->bfParams.initialCapacity);
        return &p->base;
    } catch (...) {
        return NULL;
    }
}

extern "C" void BruteForce_Free(VecSimIndex *index) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    for (auto& vectorBlock : bfIndex->vectorBlocks) {
        VectorBlock_Delete(vectorBlock);
    }
}

static void BruteForce_UpdateVector(BruteForceIndex* bfIndex, idType id, const void *vector_data) {
    // Get the vector block
    VectorBlockMember* vectorBlockMember = bfIndex->idToVectorBlockMemberMapping[id];
    VectorBlock *vectorBlock = vectorBlockMember->block;
    size_t index = vectorBlockMember->index;
    // Update vector data in the block.
    float* destinaion =  vectorBlock->vectors + (index* bfIndex->dim);
    memcpy(destinaion, vector_data, bfIndex->dim);
}

static void VectorBlock_AddVector(VectorBlock* vectorBlock, VectorBlockMember* vectorBlockMember, const void* vectorData, size_t vectorDim) {
    // Mutual point both structs on each other.
    vectorBlock->members[vectorBlock->size] = vectorBlockMember;
    vectorBlockMember->block = vectorBlock;
    vectorBlockMember->index = vectorBlock->size;

    // Copy vector data and update block size.
    memcpy(vectorBlock->vectors+(vectorBlock->size*vectorDim), vectorData, vectorDim*sizeof(float));
    vectorBlock->size++;
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
            bfIndex->count++;
        }
        else {
            id = bfIndex->count++;
        }
    }

    // See if new id is bigger than current vector count. Needs to resize the index.
    if(id > bfIndex->idToVectorBlockMemberMapping.size()) {
        bfIndex->idToVectorBlockMemberMapping.resize(bfIndex->count*2);
    }

    // Get vector block to store the vector in.
    VectorBlock* vectorBlock;
    if (bfIndex->vectorBlocks.size() == 0){
        // No vector blocks, create new one.
        vectorBlock = VectorBlock_new(bfIndex->vectorBlockSize, bfIndex->dim);
        bfIndex->vectorBlocks.push_back(vectorBlock);
    }
    else {
        // Get the last vector block.
       vectorBlock = bfIndex->vectorBlocks[bfIndex->vectorBlocks.size() - 1];
       if(vectorBlock->size == bfIndex->vectorBlockSize) {
           // Last vector block is full, create a new one.
           vectorBlock = VectorBlock_new(bfIndex->vectorBlockSize, bfIndex->dim);
           bfIndex->vectorBlocks.push_back(vectorBlock);
       }

    }

    // Create vector block membership.
    VectorBlockMember* vectorBlockMember = new VectorBlockMember();
    bfIndex->idToVectorBlockMemberMapping[id] = vectorBlockMember;
    vectorBlockMember->label = label;
    VectorBlock_AddVector(vectorBlock, vectorBlockMember, vector_data, bfIndex->dim);
    bfIndex->labelToIdLookup[label]=id;
    return true;
}

extern "C" int BruteForce_DeleteVector(VecSimIndex *index, size_t label) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    idType id;
    auto optionalId = bfIndex->labelToIdLookup.find(label);
    if(optionalId==bfIndex->labelToIdLookup.end()) {
        // Nothing to delete;
        return true;
    }
    else {
        id = optionalId->second;
    }

    size_t vectorDim = bfIndex->dim;
    // Get the vector block, and vector block member of the vector to be deleted.
    VectorBlockMember *vectorBlockMember = bfIndex->idToVectorBlockMemberMapping[id];
    VectorBlock* vectorBlock = vectorBlockMember->block;
    size_t vectorIndex = vectorBlockMember->index;

    VectorBlock* lastVectorBlock = bfIndex->vectorBlocks[bfIndex->vectorBlocks.size() - 1];
    VectorBlockMember* lastVectorBlockMember = lastVectorBlock->members[lastVectorBlock->size - 1];

    // Swap the last vector with the deleted vector;
    vectorBlock->members[vectorIndex] = lastVectorBlockMember;
    float* destination = vectorBlock->vectors+(vectorIndex*vectorDim);
    float* origin = lastVectorBlock->vectors+(lastVectorBlockMember->index*vectorDim);
    memmove(destination, origin, sizeof(float)*vectorDim );
    lastVectorBlock->size--;

    // Delete the vector block membeship
    delete vectorBlockMember;
    bfIndex->idToVectorBlockMemberMapping[id] = NULL;
    // Add deleted id to reusable ids.
    bfIndex->deletedIds.emplace(id);
    bfIndex->labelToIdLookup.erase(label);

    // If the last vector block is emtpy;
    if(lastVectorBlock->size == 0) {
        VectorBlock_Delete(lastVectorBlock);
        bfIndex->vectorBlocks.pop_back();
    }

    // Reduce index size.
    bfIndex->count--;
    return true;

}

extern "C" size_t BruteForce_Size(VecSimIndex *index) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    return bfIndex->count;
}

extern "C" VecSimQueryResult *BruteForce_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                  VecSimQueryParams *queryParams) {

        BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
        float scores[bfIndex->vectorBlockSize];
        size_t dim = bfIndex->dim;
        float upperBound = std::numeric_limits<float>::min();
        std::priority_queue<std::pair<float, labelType>, std::vector<std::pair<float, labelType>>, CompareByFirst> knn_res;
        for(auto vectorBlock : bfIndex->vectorBlocks) {
            bfIndex->distanceCalculationFunction(dim, vectorBlock, queryBlob, scores);
            for(int i =0; i < MIN(vectorBlock->size, k); i++) {
                size_t min_index = cblas_ismin(vectorBlock->size, scores, 1);
                if(knn_res.size()<k) {
                    labelType label = vectorBlock->members[min_index]->label;
                    knn_res.emplace(scores[min_index], label);
                    scores[min_index] = std::numeric_limits<float>::max();
                }
                else {
                    if(scores[min_index] <= upperBound) {
                        break;
                    }
                    else {
                        labelType label = vectorBlock->members[min_index]->label;
                        knn_res.emplace(scores[min_index], label);
                        scores[min_index] = std::numeric_limits<float>::max();
                        knn_res.pop();
                        upperBound = knn_res.top().first;
                    }
                }
            }
        }
     VecSimQueryResult *results =
            array_new_len<VecSimQueryResult>(knn_res.size(), knn_res.size());
        for (int i = knn_res.size() - 1; i >= 0; --i) {
            results[i] = VecSimQueryResult{knn_res.top().second, knn_res.top().first};
            knn_res.pop();
        }
    return results;
}

extern "C" VecSimIndexInfo BruteForce_Info(VecSimIndex *index) {
    auto idx = reinterpret_cast<BruteForceIndex *>(index);

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_BF;
    info.d = idx->dim;
    info.type = VecSimType_FLOAT32;
    return info;
}

// TODO

extern "C" VecSimQueryResult *BruteForce_DistanceQuery(VecSimIndex *index, const void *queryBlob, float distance,
                                      VecSimQueryParams queryParams);

extern "C" void BruteForce_ClearDeleted(VecSimIndex *index);


BruteForceIndex::BruteForceIndex(VecSimType vectype, VecSimMetric metric, size_t dim, size_t max_elements):
metric(metric), vecType(vecType), dim(dim), vectorBlockSize(1024*1024) {
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
    this->idToVectorBlockMemberMapping.resize(max_elements);
    if(this->metric == VecSimMetric_IP) {
        this->distanceCalculationFunction = BruteForceIndex_InternalProduct;
    } else if(this->metric == VecSimMetric_L2) {
        this->distanceCalculationFunction = BruteForceIndex_L2;
    }
}
