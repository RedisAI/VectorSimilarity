#include "brute_force.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <cstring>

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
                                      

}

extern "C" VecSimIndexInfo BruteForce_Info(VecSimIndex *index);

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
}
