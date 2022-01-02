
#include "brute_force.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/bf_batch_iterator.h"

#include <memory>
#include <cstring>
#include <queue>

using namespace std;

/******************** Ctor / Dtor **************/
BruteForceIndex::BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : 
    VecSimIndex(allocator),
    dim(params->dim),
    vecType(params->type),
    metric(params->metric),
    labelToIdLookup(allocator), idToVectorBlockMemberMapping(allocator), deletedIds(allocator),
    vectorBlocks(allocator),
    vectorBlockSize(params->blockSize ? params->blockSize : BF_DEFAULT_BLOCK_SIZE),
    count(0),
    space(params->metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new (allocator)
                                                           L2Space(params->dim, allocator))
                : static_cast<SpaceInterface<float> *>(
                      new (allocator) InnerProductSpace(params->dim, allocator))) {
    this->idToVectorBlockMemberMapping.resize(params->initialCapacity);
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

    float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, vector_data, this->dim * sizeof(float));
        float_vector_normalize(normalized_data, this->dim);
        vector_data = normalized_data;
    }

    idType id = 0;
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
    if (id >= this->idToVectorBlockMemberMapping.size()) {
        this->idToVectorBlockMemberMapping.resize(this->count * 2);
    }

    // Get vector block to store the vector in.
    VectorBlock *vectorBlock;
    if (this->vectorBlocks.size() == 0) {
        // No vector blocks, create new one.
        vectorBlock =
            new (this->allocator) VectorBlock(this->vectorBlockSize, this->dim, this->allocator);
        this->vectorBlocks.push_back(vectorBlock);
    } else {
        // Get the last vector block.
        vectorBlock = this->vectorBlocks[this->vectorBlocks.size() - 1];
        if (vectorBlock->getLength() == this->vectorBlockSize) {
            // Last vector block is full, create a new one.
            vectorBlock = new (this->allocator)
                VectorBlock(this->vectorBlockSize, this->dim, this->allocator);
            this->vectorBlocks.push_back(vectorBlock);
        }
    }

    // Create vector block membership.
    VectorBlockMember *vectorBlockMember = new (this->allocator) VectorBlockMember(this->allocator);
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
        lastVectorBlock->getMember(lastVectorBlock->getLength() - 1);

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
    if (lastVectorBlock->getLength() == 0) {
        delete lastVectorBlock;
        this->vectorBlocks.pop_back();
    }

    // Reduce index size.
    this->count--;
    return true;
}

size_t BruteForceIndex::indexSize() const { return this->count; }

VecSimQueryResult_List BruteForceIndex::topKQuery(const void *queryBlob, size_t k,
                                                  VecSimQueryParams *queryParams) {

    float normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(float));
        float_vector_normalize(normalized_blob, this->dim);
        queryBlob = normalized_blob;
    }

    float upperBound = std::numeric_limits<float>::lowest();
    vecsim_stl::max_priority_queue<pair<float, labelType>> TopCandidates(this->allocator);
    // For every block, compute its vectors scores and update the Top candidates max heap
    for (auto vectorBlock : this->vectorBlocks) {
        size_t block_size = vectorBlock->getLength();
        vecsim_stl::vector<float> scores(block_size, this->allocator);
        for (size_t i = 0; i < block_size; i++) {
            scores[i] = this->dist_func(vectorBlock->getVector(i), queryBlob, &this->dim);
        }
        for (int i = 0; i < scores.size(); i++) {
            // Always choose the current candidate if we have less than k.
            if (TopCandidates.size() < k) {
                TopCandidates.emplace(scores[i], vectorBlock->getMember(i)->label);
                upperBound = TopCandidates.top().first;
            } else {
                // Otherwise, try greedily to improve the top candidates with a vector that
                // has a better score than the one that has the worst score until now.
                if (scores[i] >= upperBound) {
                    continue;
                } else {
                    TopCandidates.emplace(scores[i], vectorBlock->getMember(i)->label);
                    TopCandidates.pop();
                    upperBound = TopCandidates.top().first;
                }
            }
        }
    }
    auto *results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(results[i], TopCandidates.top().first);
        TopCandidates.pop();
    }
    return results;
}

VecSimIndexInfo BruteForceIndex::info() {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_BF;
    info.bfInfo.dim = this->dim;
    info.bfInfo.type = this->vecType;
    info.bfInfo.metric = this->metric;
    info.bfInfo.indexSize = this->count;
    info.bfInfo.blockSize = this->vectorBlockSize;
    info.bfInfo.memory = this->allocator->getAllocationSize();
    return info;
}

VecSimInfoIterator *BruteForceIndex::infoIterator() {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed;
    size_t numberOfInfoFields = 7;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::ALGORITHM_STRING,
                                .fieldType = INFOFIELD_STRING,
                                .stringValue = VecSimAlgo_ToString(info.algo)});
    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::TYPE_STRING,
                                .fieldType = INFOFIELD_STRING,
                                .stringValue = VecSimType_ToString(info.bfInfo.type)});
    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.bfInfo.dim});
    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::METRIC_STRING,
                                .fieldType = INFOFIELD_STRING,
                                .stringValue = VecSimMetric_ToString(info.bfInfo.metric)});
    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.bfInfo.indexSize});
    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.bfInfo.blockSize});
    infoIterator->addInfoField({.fieldName = VecSimCommonStrings::MEMORY_STRING,
                                .fieldType = INFOFIELD_UINT64,
                                .uintegerValue = info.bfInfo.memory});

    return infoIterator;
}

VecSimBatchIterator *BruteForceIndex::newBatchIterator(const void *queryBlob) {
    return new (this->allocator) BF_BatchIterator(queryBlob, this, this->allocator);
}
