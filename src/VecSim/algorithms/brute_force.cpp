#include "brute_force.h"
#include <unordered_map>
<<<<<<< HEAD
=======
#include <memory>
>>>>>>> origin/main
#include <set>
#include <vector>
#include <cstring>
#include <queue>
<<<<<<< HEAD
#include "cblas.h"
#include <limits>
#include "VecSim/utils/arr_cpp.h"
=======
#include <limits>
#include "VecSim/spaces/space_interface.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/utils/arr_cpp.h"
#include <iostream>
>>>>>>> origin/main

using namespace std;

typedef size_t labelType;
typedef size_t idType;

// Predeclaration

struct VectorBlock;

<<<<<<< HEAD
struct VectorBlockMember{
    size_t index;
    VectorBlock* block;
=======
struct VectorBlockMember {
    size_t index;
    VectorBlock *block;
>>>>>>> origin/main
    labelType label;
};

struct VectorBlock {
<<<<<<< HEAD
    size_t size;
    VectorBlockMember** members;
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
    std::vector<VectorBlockMember*> idToVectorBlockMemberMapping;
    std::set<idType> deletedIds;
    std::vector<VectorBlock*> vectorBlocks;
    size_t vectorBlockSize;
=======
    VectorBlock(size_t blockSize, size_t vectorSize) {
        this->size = 0;
        this->members = new VectorBlockMember *[blockSize];
        this->vectors = new float[blockSize * vectorSize];
    }
    size_t size;
    VectorBlockMember **members;
    float *vectors;

    ~VectorBlock() {
        for (size_t i = 0; i < this->size; i++) {
            delete members[i];
        }
        delete[] members;
        delete[] vectors;
    }
};
struct BruteForceIndex {
    BruteForceIndex(VecSimType vectype, VecSimMetric metric, size_t dim, size_t max_elements,
                    size_t blockSize);

    // Meta data
    VecSimIndex base;
    // TODO: support cosine, angular

    idType count;
    std::unordered_map<labelType, idType> labelToIdLookup;
    std::vector<VectorBlockMember *> idToVectorBlockMemberMapping;
    std::set<idType> deletedIds;
    std::vector<VectorBlock *> vectorBlocks;
    size_t vectorBlockSize;
    unique_ptr<SpaceInterface<float>> space;
    DISTFUNC<float> dist_func;
>>>>>>> origin/main
};

struct CompareByFirst {
    constexpr bool operator()(std::pair<float, labelType> const &a,
<<<<<<< HEAD
                                std::pair<float, labelType> const &b) const noexcept {
=======
                              std::pair<float, labelType> const &b) const noexcept {
>>>>>>> origin/main
        return a.first < b.first;
    }
};

<<<<<<< HEAD
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
=======
extern "C" VecSimIndex *BruteForce_New(const VecSimParams *params) {
    try {
        auto p = new BruteForceIndex(
            params->type, params->metric, params->size, params->bfParams.initialCapacity,
            params->bfParams.blockSize ? params->bfParams.blockSize : BF_DEFAULT_BLOCK_SIZE);
>>>>>>> origin/main
        return &p->base;
    } catch (...) {
        return NULL;
    }
}

extern "C" void BruteForce_Free(VecSimIndex *index) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
<<<<<<< HEAD
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
=======
    for (auto &vectorBlock : bfIndex->vectorBlocks) {
        delete vectorBlock;
    }
    delete bfIndex;
}

static void BruteForce_UpdateVector(BruteForceIndex *bfIndex, idType id, const void *vector_data) {
    // Get the vector block
    VectorBlockMember *vectorBlockMember = bfIndex->idToVectorBlockMemberMapping[id];
    VectorBlock *vectorBlock = vectorBlockMember->block;
    size_t index = vectorBlockMember->index;
    // Update vector data in the block.
    float *destinaion = vectorBlock->vectors + (index * bfIndex->base.dim);
    memcpy(destinaion, vector_data, bfIndex->base.dim);
}

static void VectorBlock_AddVector(VectorBlock *vectorBlock, VectorBlockMember *vectorBlockMember,
                                  const void *vectorData, size_t vectorDim) {
>>>>>>> origin/main
    // Mutual point both structs on each other.
    vectorBlock->members[vectorBlock->size] = vectorBlockMember;
    vectorBlockMember->block = vectorBlock;
    vectorBlockMember->index = vectorBlock->size;

    // Copy vector data and update block size.
<<<<<<< HEAD
    memcpy(vectorBlock->vectors+(vectorBlock->size*vectorDim), vectorData, vectorDim*sizeof(float));
=======
    memcpy(vectorBlock->vectors + (vectorBlock->size * vectorDim), vectorData,
           vectorDim * sizeof(float));
>>>>>>> origin/main
    vectorBlock->size++;
}

extern "C" int BruteForce_AddVector(VecSimIndex *index, const void *vector_data, size_t label) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);

<<<<<<< HEAD
    idType id;
=======
    idType id = 0;
>>>>>>> origin/main
    bool update = false;
    auto optionalID = bfIndex->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != bfIndex->labelToIdLookup.end()) {
        id = optionalID->second;
        BruteForce_UpdateVector(bfIndex, id, vector_data);
        return true;
<<<<<<< HEAD
    }
    else {
        // Try re-use deleted id.
        if(bfIndex->deletedIds.size()!=0) {
            id = *bfIndex->deletedIds.begin();
            bfIndex->deletedIds.erase(bfIndex->deletedIds.begin());
        }
        else {
=======
    } else {
        // Try re-use deleted id.
        if (bfIndex->deletedIds.size() != 0) {
            id = *bfIndex->deletedIds.begin();
            bfIndex->deletedIds.erase(bfIndex->deletedIds.begin());
            bfIndex->count++;
        } else {
>>>>>>> origin/main
            id = bfIndex->count++;
        }
    }

    // See if new id is bigger than current vector count. Needs to resize the index.
<<<<<<< HEAD
    if(id > bfIndex->idToVectorBlockMemberMapping.size()) {
        bfIndex->idToVectorBlockMemberMapping.resize(bfIndex->count*2);
    }

    // Get vector block to store the vector in.
    VectorBlock* vectorBlock;
    if (bfIndex->vectorBlocks.size() == 0){
        // No vector blocks, create new one.
        vectorBlock = new VectorBlock();
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
=======
    if (id > bfIndex->idToVectorBlockMemberMapping.size()) {
        bfIndex->idToVectorBlockMemberMapping.resize(bfIndex->count * 2);
    }

    // Get vector block to store the vector in.
    VectorBlock *vectorBlock;
    if (bfIndex->vectorBlocks.size() == 0) {
        // No vector blocks, create new one.
        vectorBlock = new VectorBlock(bfIndex->vectorBlockSize, bfIndex->base.dim);
        bfIndex->vectorBlocks.push_back(vectorBlock);
    } else {
        // Get the last vector block.
        vectorBlock = bfIndex->vectorBlocks[bfIndex->vectorBlocks.size() - 1];
        if (vectorBlock->size == bfIndex->vectorBlockSize) {
            // Last vector block is full, create a new one.
            vectorBlock = new VectorBlock(bfIndex->vectorBlockSize, bfIndex->base.dim);
            bfIndex->vectorBlocks.push_back(vectorBlock);
        }
    }

    // Create vector block membership.
    VectorBlockMember *vectorBlockMember = new VectorBlockMember();
    bfIndex->idToVectorBlockMemberMapping[id] = vectorBlockMember;
    vectorBlockMember->label = label;
    VectorBlock_AddVector(vectorBlock, vectorBlockMember, vector_data, bfIndex->base.dim);
    bfIndex->labelToIdLookup.emplace(label, id);
>>>>>>> origin/main
    return true;
}

extern "C" int BruteForce_DeleteVector(VecSimIndex *index, size_t label) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    idType id;
    auto optionalId = bfIndex->labelToIdLookup.find(label);
<<<<<<< HEAD
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
=======
    if (optionalId == bfIndex->labelToIdLookup.end()) {
        // Nothing to delete;
        return true;
    } else {
        id = optionalId->second;
    }

    size_t vectorDim = bfIndex->base.dim;
    // Get the vector block, and vector block member of the vector to be deleted.
    VectorBlockMember *vectorBlockMember = bfIndex->idToVectorBlockMemberMapping[id];
    VectorBlock *vectorBlock = vectorBlockMember->block;
    size_t vectorIndex = vectorBlockMember->index;

    VectorBlock *lastVectorBlock = bfIndex->vectorBlocks[bfIndex->vectorBlocks.size() - 1];
    VectorBlockMember *lastVectorBlockMember = lastVectorBlock->members[lastVectorBlock->size - 1];

    // Swap the last vector with the deleted vector;
    vectorBlock->members[vectorIndex] = lastVectorBlockMember;
    float *destination = vectorBlock->vectors + (vectorIndex * vectorDim);
    float *origin = lastVectorBlock->vectors + (lastVectorBlockMember->index * vectorDim);
    memmove(destination, origin, sizeof(float) * vectorDim);
    lastVectorBlock->size--;

    // Delete the vector block membership
>>>>>>> origin/main
    delete vectorBlockMember;
    bfIndex->idToVectorBlockMemberMapping[id] = NULL;
    // Add deleted id to reusable ids.
    bfIndex->deletedIds.emplace(id);
<<<<<<< HEAD

    // If the last vector block is emtpy;
    if(lastVectorBlock->size == 0) {
        VectorBlock_Delete(lastVectorBlock);
=======
    bfIndex->labelToIdLookup.erase(label);

    // If the last vector block is emtpy;
    if (lastVectorBlock->size == 0) {
        delete lastVectorBlock;
>>>>>>> origin/main
        bfIndex->vectorBlocks.pop_back();
    }

    // Reduce index size.
    bfIndex->count--;
    return true;
<<<<<<< HEAD

=======
>>>>>>> origin/main
}

extern "C" size_t BruteForce_Size(VecSimIndex *index) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    return bfIndex->count;
}

<<<<<<< HEAD
extern "C" VecSimQueryResult *BruteForce_TopKQuery(VecSimIndex *index, const void *queryBlob, size_t k,
                                  VecSimQueryParams *queryParams) {

        BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
        float scores[bfIndex->vectorBlockSize];
        size_t dim = bfIndex->dim;
        float lowerBound = std::numeric_limits<float>::max();
        std::priority_queue<std::pair<float, labelType>, std::vector<std::pair<float, labelType>>, CompareByFirst> knn_res;
        for(auto vectorBlock : bfIndex->vectorBlocks) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, vectorBlock->size, dim, 1, vectorBlock->vectors,  dim, (const float *)queryBlob, 1 , 0 , scores, 1);
            for(int i =0; i < MIN(vectorBlock->size, k); i++) {
                size_t max_index = cblas_isamax(vectorBlock->size, scores, 1);
                if(knn_res.size()<k) {
                    labelType label = vectorBlock->members[max_index]->label;
                    knn_res.emplace(scores[max_index], label);
                    scores[max_index] = std::numeric_limits<float>::min();
                }
                else {
                    if(scores[max_index] >= lowerBound) {
                        break;
                    }
                    else {
                        labelType label = vectorBlock->members[max_index]->label;
                        knn_res.emplace(scores[max_index], label);
                        scores[max_index] = std::numeric_limits<float>::min();
                        knn_res.pop();
                        lowerBound = knn_res.top().first;
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
=======
extern "C" VecSimQueryResult *BruteForce_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                   size_t k, VecSimQueryParams *queryParams) {

    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    size_t dim = bfIndex->base.dim;
    float upperBound = std::numeric_limits<float>::min();
    std::priority_queue<std::pair<float, labelType>, std::vector<std::pair<float, labelType>>,
                        CompareByFirst>
        knn_res;
    for (auto vectorBlock : bfIndex->vectorBlocks) {
        float scores[bfIndex->vectorBlockSize];
        for (size_t i = 0; i < vectorBlock->size; i++) {
            scores[i] = bfIndex->dist_func(vectorBlock->vectors + (i * dim), queryBlob, &dim);
        }
        size_t vec_count = vectorBlock->size;
        for (int i = 0; i < vec_count; i++) {

            if (knn_res.size() < k) {
                labelType label = vectorBlock->members[i]->label;
                knn_res.emplace(scores[i], label);
                upperBound = knn_res.top().first;
            } else {
                if (scores[i] >= upperBound) {
                    continue;
                } else {
                    labelType label = vectorBlock->members[i]->label;
                    knn_res.emplace(scores[i], label);
                    knn_res.pop();
                    upperBound = knn_res.top().first;
                }
            }
        }
    }
    VecSimQueryResult *results = array_new_len<VecSimQueryResult>(knn_res.size(), knn_res.size());
    for (int i = knn_res.size() - 1; i >= 0; --i) {
        results[i] = VecSimQueryResult{knn_res.top().second, knn_res.top().first};
        knn_res.pop();
    }
>>>>>>> origin/main
    return results;
}

extern "C" VecSimIndexInfo BruteForce_Info(VecSimIndex *index) {
    auto idx = reinterpret_cast<BruteForceIndex *>(index);

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_BF;
<<<<<<< HEAD
    info.d = idx->dim;
    info.type = VecSimType_FLOAT32;
=======
    info.d = index->dim;
    info.type = index->vecType;
    info.metric = index->metric;
    info.bfInfo.indexSize = idx->count;
    info.bfInfo.blockSize = idx->vectorBlockSize;
>>>>>>> origin/main
    return info;
}

// TODO

<<<<<<< HEAD
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
=======
extern "C" VecSimQueryResult *BruteForce_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                                       float distance,
                                                       VecSimQueryParams queryParams);

extern "C" void BruteForce_ClearDeleted(VecSimIndex *index);

BruteForceIndex::BruteForceIndex(VecSimType vectype, VecSimMetric metric, size_t dim,
                                 size_t max_elements, size_t blockSize)
    : vectorBlockSize(blockSize), count(0),
      space(metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new L2Space(dim))
                : static_cast<SpaceInterface<float> *>(new InnerProductSpace(dim))) {
    this->base = VecSimIndex{
        AddFn : BruteForce_AddVector,
        DeleteFn : BruteForce_DeleteVector,
        SizeFn : BruteForce_Size,
        TopKQueryFn : BruteForce_TopKQuery,
        DistanceQueryFn : NULL,
        ClearDeletedFn : NULL,
        FreeFn : BruteForce_Free,
        InfoFn : BruteForce_Info,
        dim : dim,
        vecType : vectype,
        metric : metric
    };
    this->idToVectorBlockMemberMapping.resize(max_elements);
    this->dist_func = this->space.get()->get_dist_func();
>>>>>>> origin/main
}
