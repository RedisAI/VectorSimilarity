#include "brute_force.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <cstring>
#include <queue>
#include "OpenBLAS/cblas.h"
#include <limits>
#include "VecSim/utils/arr_cpp.h"
#include <iostream>

using namespace std;

typedef size_t labelType;
typedef size_t idType;

// Predeclaration

struct VectorBlock;

struct VectorBlockMember {
    size_t index;
    VectorBlock *block;
    labelType label;
};

struct VectorBlock {
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

typedef void (*DistanceCalculateFunction)(size_t dim, VectorBlock *vectorBlock,
                                          const void *queryBlob, float *scores);

void BruteForceIndex_InternalProduct(size_t dim, VectorBlock *vectorBlock, const void *queryBlob,
                                     float *scores) {

    // Calculate AxV internal product into tmp_scores.
    cblas_sgemv(CblasRowMajor, CblasNoTrans, vectorBlock->size, dim, -1, vectorBlock->vectors, dim,
                (const float *)queryBlob, 1, 1, scores, 1);
}

void BruteForceIndex_L2(size_t dim, VectorBlock *vectorBlock, const void *queryBlob,
                        float *scores) {
    float tmp_vector[dim];
    for (size_t i = 0; i < vectorBlock->size; i++) {
        cblas_scopy(dim, vectorBlock->vectors + (i * dim), 1, tmp_vector, 1);
        cblas_saxpy(dim, -1.0f, (const float *)queryBlob, 1, tmp_vector, 1);
        scores[i] = cblas_sdot(dim, tmp_vector, 1, tmp_vector, 1);
    }
}

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

    DistanceCalculateFunction distanceCalculationFunction;
};

struct CompareByFirst {
    constexpr bool operator()(std::pair<float, labelType> const &a,
                              std::pair<float, labelType> const &b) const noexcept {
        return a.first < b.first;
    }
};

extern "C" VecSimIndex *BruteForce_New(const VecSimParams *params) {
    try {
        auto p = new BruteForceIndex(
            params->type, params->metric, params->size, params->bfParams.initialCapacity,
            params->bfParams.blockSize ? params->bfParams.blockSize : BF_DEFAULT_BLOCK_SIZE);
        return &p->base;
    } catch (...) {
        return NULL;
    }
}

extern "C" void BruteForce_Free(VecSimIndex *index) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
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
    // Mutual point both structs on each other.
    vectorBlock->members[vectorBlock->size] = vectorBlockMember;
    vectorBlockMember->block = vectorBlock;
    vectorBlockMember->index = vectorBlock->size;

    // Copy vector data and update block size.
    memcpy(vectorBlock->vectors + (vectorBlock->size * vectorDim), vectorData,
           vectorDim * sizeof(float));
    vectorBlock->size++;
}

extern "C" int BruteForce_AddVector(VecSimIndex *index, const void *vector_data, size_t label) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);

    idType id = 0;
    bool update = false;
    auto optionalID = bfIndex->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != bfIndex->labelToIdLookup.end()) {
        id = optionalID->second;
        BruteForce_UpdateVector(bfIndex, id, vector_data);
        return true;
    } else {
        // Try re-use deleted id.
        if (bfIndex->deletedIds.size() != 0) {
            id = *bfIndex->deletedIds.begin();
            bfIndex->deletedIds.erase(bfIndex->deletedIds.begin());
            bfIndex->count++;
        } else {
            id = bfIndex->count++;
        }
    }

    // See if new id is bigger than current vector count. Needs to resize the index.
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
    return true;
}

extern "C" int BruteForce_DeleteVector(VecSimIndex *index, size_t label) {
    BruteForceIndex *bfIndex = reinterpret_cast<BruteForceIndex *>(index);
    idType id;
    auto optionalId = bfIndex->labelToIdLookup.find(label);
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
    delete vectorBlockMember;
    bfIndex->idToVectorBlockMemberMapping[id] = NULL;
    // Add deleted id to reusable ids.
    bfIndex->deletedIds.emplace(id);
    bfIndex->labelToIdLookup.erase(label);

    // If the last vector block is emtpy;
    if (lastVectorBlock->size == 0) {
        delete lastVectorBlock;
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
        std::fill_n(scores, bfIndex->vectorBlockSize, 1.0);
        bfIndex->distanceCalculationFunction(dim, vectorBlock, queryBlob, scores);
        for (int i = 0; i < MIN(vectorBlock->size, k); i++) {
            size_t min_index = cblas_ismin(vectorBlock->size, scores, 1);
            if (knn_res.size() < k) {
                labelType label = vectorBlock->members[min_index]->label;
                knn_res.emplace(scores[min_index], label);
                scores[min_index] = std::numeric_limits<float>::max();
                upperBound = knn_res.top().first;
            } else {
                if (scores[min_index] >= upperBound) {
                    break;
                } else {
                    labelType label = vectorBlock->members[min_index]->label;
                    knn_res.emplace(scores[min_index], label);
                    scores[min_index] = std::numeric_limits<float>::max();
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
    return results;
}

extern "C" VecSimIndexInfo BruteForce_Info(VecSimIndex *index) {
    auto idx = reinterpret_cast<BruteForceIndex *>(index);

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_BF;
    info.d = index->dim;
    info.type = index->vecType;
    info.metric = index->metric;
    info.bfInfo.indexSize = idx->count;
    info.bfInfo.blockSize = idx->vectorBlockSize;
    return info;
}

// TODO

extern "C" VecSimQueryResult *BruteForce_DistanceQuery(VecSimIndex *index, const void *queryBlob,
                                                       float distance,
                                                       VecSimQueryParams queryParams);

extern "C" void BruteForce_ClearDeleted(VecSimIndex *index);

BruteForceIndex::BruteForceIndex(VecSimType vectype, VecSimMetric metric, size_t dim,
                                 size_t max_elements, size_t blockSize)
    : vectorBlockSize(blockSize), count(0) {
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
    if (this->base.metric == VecSimMetric_IP || this->base.metric == VecSimMetric_Cosine) {
        this->distanceCalculationFunction = BruteForceIndex_InternalProduct;
    } else if (this->base.metric == VecSimMetric_L2) {
        this->distanceCalculationFunction = BruteForceIndex_L2;
    }
}
