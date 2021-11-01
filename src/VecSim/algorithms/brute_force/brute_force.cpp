
#include "brute_force.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/query_result_struct.h"

#include <memory>
#include <set>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <queue>

using namespace std;

// TODO: unify this with HNSW
struct CompareByFirst {
    constexpr bool operator()(std::pair<float, labelType> const &a,
                              std::pair<float, labelType> const &b) const noexcept {
        return a.first < b.first;
    }
};

/******************** Ctor / Dtor **************/
BruteForceIndex::BruteForceIndex(const VecSimParams *params)
    : VecSimIndex(params), vectorBlockSize(params->bfParams.blockSize ? params->bfParams.blockSize
                                                                      : BF_DEFAULT_BLOCK_SIZE),
      count(0),
      space(params->metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new L2Space(params->size))
                : static_cast<SpaceInterface<float> *>(new InnerProductSpace(params->size))) {
    this->idToVectorBlockMemberMapping.resize(params->bfParams.initialCapacity);
    this->dist_func = this->space->get_dist_func();
}

BruteForceIndex::~BruteForceIndex() {
    for (auto &vectorBlock : this->vectorBlocks) {
        delete vectorBlock;
    }
}

/******************** Implementation **************/
void BruteForceIndex::updateVector(idType id, const void *vector_data) {
    // Get the vector block
    VectorBlockMember *vectorBlockMember = this->idToVectorBlockMemberMapping[id];
    VectorBlock *vectorBlock = vectorBlockMember->block;
    size_t index = vectorBlockMember->index;
    // Update vector data in the block.
    float *destinaion = vectorBlock->getVector(index);
    memcpy(destinaion, vector_data, this->dim);
}

int BruteForceIndex::addVector(const void *vector_data, size_t label) {

    idType id = 0;
    bool update = false;
    auto optionalID = this->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != this->labelToIdLookup.end()) {
        id = optionalID->second;
        updateVector(id, vector_data);
        return true;
    } else {
        // Try re-use deleted id.
        if (this->deletedIds.size() != 0) {
            id = *this->deletedIds.begin();
            this->deletedIds.erase(this->deletedIds.begin());
            this->count++;
        } else {
            id = this->count++;
        }
    }

    // See if new id is bigger than current vector count. Needs to resize the index.
    if (id > this->idToVectorBlockMemberMapping.size()) {
        this->idToVectorBlockMemberMapping.resize(this->count * 2);
    }

    // Get vector block to store the vector in.
    VectorBlock *vectorBlock;
    if (this->vectorBlocks.size() == 0) {
        // No vector blocks, create new one.
        vectorBlock = new VectorBlock(this->vectorBlockSize, this->dim);
        this->vectorBlocks.push_back(vectorBlock);
    } else {
        // Get the last vector block.
        vectorBlock = this->vectorBlocks[this->vectorBlocks.size() - 1];
        if (vectorBlock->getSize() == this->vectorBlockSize) {
            // Last vector block is full, create a new one.
            vectorBlock = new VectorBlock(this->vectorBlockSize, this->dim);
            this->vectorBlocks.push_back(vectorBlock);
        }
    }

    // Create vector block membership.
    VectorBlockMember *vectorBlockMember = new VectorBlockMember();
    this->idToVectorBlockMemberMapping[id] = vectorBlockMember;
    vectorBlockMember->label = label;
    vectorBlock->addVector(vectorBlockMember, vector_data);
    this->labelToIdLookup.emplace(label, id);
    return true;
}

int BruteForceIndex::deleteVector(size_t label) {
    idType id;
    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        // Nothing to delete;
        return true;
    } else {
        id = optionalId->second;
    }

    // Get the vector block, and vector block member of the vector to be deleted.
    VectorBlockMember *vectorBlockMember = this->idToVectorBlockMemberMapping[id];
    VectorBlock *vectorBlock = vectorBlockMember->block;
    size_t vectorIndex = vectorBlockMember->index;

    VectorBlock *lastVectorBlock = this->vectorBlocks[this->vectorBlocks.size() - 1];
    VectorBlockMember *lastVectorBlockMember =
        lastVectorBlock->getMember(lastVectorBlock->getSize() - 1);

    // Swap the last vector with the deleted vector;
    vectorBlock->setMember(vectorIndex, lastVectorBlockMember);
    lastVectorBlockMember->block = vectorBlock;

    float *destination = vectorBlock->getVector(vectorIndex);
    float *origin = lastVectorBlock->removeAndFetchVector();
    memmove(destination, origin, sizeof(float) * this->dim);

    // Delete the vector block membership
    delete vectorBlockMember;
    this->idToVectorBlockMemberMapping[id] = NULL;
    // Add deleted id to reusable ids.
    this->deletedIds.emplace(id);
    this->labelToIdLookup.erase(label);

    // If the last vector block is emtpy;
    if (lastVectorBlock->getSize() == 0) {
        delete lastVectorBlock;
        this->vectorBlocks.pop_back();
    }

    // Reduce index size.
    this->count--;
    return true;
}

size_t BruteForceIndex::indexSize() { return this->count; }

VecSimQueryResult_List BruteForceIndex::topKQuery(const void *queryBlob, size_t k,
                                                  VecSimQueryParams *queryParams) {

    float upperBound = std::numeric_limits<float>::min();
    std::priority_queue<std::pair<float, labelType>, std::vector<std::pair<float, labelType>>,
                        CompareByFirst>
        knn_res;
    for (auto vectorBlock : this->vectorBlocks) {
        float scores[this->vectorBlockSize];
        for (size_t i = 0; i < vectorBlock->getSize(); i++) {
            scores[i] = this->dist_func(vectorBlock->getVector(i), queryBlob, &dim);
        }
        size_t vec_count = vectorBlock->getSize();
        for (int i = 0; i < vec_count; i++) {

            if (knn_res.size() < k) {
                labelType label = vectorBlock->getMember(i)->label;
                knn_res.emplace(scores[i], label);
                upperBound = knn_res.top().first;
            } else {
                if (scores[i] >= upperBound) {
                    continue;
                } else {
                    labelType label = vectorBlock->getMember(i)->label;
                    knn_res.emplace(scores[i], label);
                    knn_res.pop();
                    upperBound = knn_res.top().first;
                }
            }
        }
    }
    auto *results = array_new_len<VecSimQueryResult>(knn_res.size(), knn_res.size());
    for (int i = (int)knn_res.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(results[i], knn_res.top().second);
        VecSimQueryResult_SetScore(results[i], knn_res.top().first);
        knn_res.pop();
    }
    return results;
}

VecSimIndexInfo BruteForceIndex::info() {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_BF;
    info.d = this->dim;
    info.type = this->vecType;
    info.metric = this->metric;
    info.bfInfo.indexSize = this->count;
    info.bfInfo.blockSize = this->vectorBlockSize;
    return info;
}

VecSimBatchIterator *BruteForceIndex::newBatchIterator(const void *queryBlob) { return nullptr; }
