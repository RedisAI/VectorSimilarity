
#include "brute_force.h"
#include "VecSim/spaces/L2_space.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/bf_batch_iterator.h"

#include <memory>
#include <cstring>
#include <queue>
#include <cassert>
#include <cmath>

using namespace std;

/******************** Ctor / Dtor **************/
BruteForceIndex::BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndex(allocator), dim(params->dim), vecType(params->type), metric(params->metric),
      labelToIdLookup(allocator), idToLabelMapping(allocator), vectorBlocks(allocator),
      vectorBlockSize(params->blockSize ? params->blockSize : DEFAULT_BLOCK_SIZE), count(0),
      space(params->metric == VecSimMetric_L2
                ? static_cast<SpaceInterface<float> *>(new (allocator)
                                                           L2Space(params->dim, allocator))
                : static_cast<SpaceInterface<float> *>(
                      new (allocator) InnerProductSpace(params->dim, allocator))),
      last_mode(EMPTY_MODE) {
    this->idToLabelMapping.resize(params->initialCapacity);
    this->dist_func = this->space->get_dist_func();
}

BruteForceIndex::~BruteForceIndex() {
    for (auto &vectorBlock : this->vectorBlocks) {
        delete vectorBlock;
    }
}

/******************** Implementation **************/
size_t BruteForceIndex::estimateInitialSize(const BFParams *params) {
    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + sizeof(BruteForceIndex) + sizeof(size_t);
    est += (params->metric == VecSimMetric_L2 ? sizeof(L2Space) : sizeof(InnerProductSpace)) +
           sizeof(size_t);
    // Parameters related part.
    est += params->initialCapacity * sizeof(decltype(idToLabelMapping)::value_type) +
           sizeof(size_t);

    return est;
}

size_t BruteForceIndex::estimateElementMemory(const BFParams *params) {
    return params->dim * sizeof(float) + sizeof(labelType);
}

void BruteForceIndex::updateVector(idType id, const void *vector_data) {
    
    // TODO Get the vector block
    VectorBlock *vectorBlock = getVectorBlock(id);
    size_t index = getVectorRelativeIndex(id);

    // Update vector data in the block.
    vectorBlock->updateVector(index, vector_data);
}



int BruteForceIndex::addVector(const void *vector_data, size_t label) {

    float normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, vector_data, this->dim * sizeof(float));
        float_vector_normalize(normalized_data, this->dim);
        vector_data = normalized_data;
    }

    
    auto optionalID = this->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != this->labelToIdLookup.end()) {
        idType id = optionalID->second;
        updateVector(id, vector_data);
        return true;
    }

    //give the vector new id
	idType id = count;

    // Get vector block to store the vector in.

    //if vectorBlocks vector is empty ||last_vector_block is full create a new block
    if(count % vectorBlockSize == 0) {
        VectorBlock *new_vectorBlock = new (this->allocator)
            VectorBlock(this->vectorBlockSize, this->dim, this->allocator);
        this->vectorBlocks.push_back(new_vectorBlock);
    }

    // get the last vectors block
    VectorBlock *vectorBlock = this->vectorBlocks.back();

    assert(vectorBlock == vectorBlocks[count / vectorBlockSize]);

    // if idToLabelMapping is full,
	// resize and align idToLabelMapping by vectorBlockSize
    size_t idToLabelMapping_size =  this->idToLabelMapping.size();

    if (count >= idToLabelMapping_size) {
        size_t extra_space_to_free = curr_capacity % block_size;

        this->idToLabelMapping.resize(idToLabelMapping_size + vectorBlockSize, 0);
    }







	//add label to idToLabelMapping
	idToLabelMapping[id] = label;

    
    // Create vector block membership.

    this->idToLabelMapping[id] = label;
    vectorBlock->addVector(vectorBlockMember, vector_data);
    this->labelToIdLookup.emplace(label, id);


    //increase count
    ++count;
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
    VectorBlockMember *vectorBlockMember = this->idToLabelMapping[id];
    VectorBlock *vectorBlock = vectorBlockMember->block;
    size_t vectorIndex = vectorBlockMember->index;

    VectorBlock *lastVectorBlock = this->vectorBlocks[this->vectorBlocks.size() - 1];
    VectorBlockMember *lastVectorBlockMember =
        lastVectorBlock->getMember(lastVectorBlock->getLength() - 1);

    // Swap the last vector with the deleted vector;
    vectorBlock->setMember(vectorIndex, lastVectorBlockMember);

    float *destination = vectorBlock->getVector(vectorIndex);
    float *origin = lastVectorBlock->removeAndFetchVector();
    memmove(destination, origin, sizeof(float) * this->dim);

    // Delete the vector block membership
    delete vectorBlockMember;
    this->idToLabelMapping[id] = NULL;
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

double BruteForceIndex::getDistanceFrom(size_t label, const void *vector_data) {
    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = optionalId->second;
    VectorBlockMember *vector_index = this->idToLabelMapping[id];

    return this->dist_func(vector_index->block->getVector(vector_index->index), vector_data,
                           &this->dim);
}

size_t BruteForceIndex::indexSize() const { return this->count; }

// Compute the score for every vector in the block by using the given distance function.
vecsim_stl::vector<float> BruteForceIndex::computeBlockScores(VectorBlock *block,
                                                              const void *queryBlob,
                                                              void *timeoutCtx,
                                                              VecSimQueryResult_Code *rc) const {
    size_t len = block->getLength();
    vecsim_stl::vector<float> scores(len, this->allocator);
    for (size_t i = 0; i < len; i++) {
        if (__builtin_expect(VecSimIndex::timeoutCallback(timeoutCtx), 0)) {
            *rc = VecSim_QueryResult_TimedOut;
            return scores;
        }
        scores[i] = this->dist_func(block->getVector(i), queryBlob, &this->dim);
    }
    *rc = VecSim_QueryResult_OK;
    return scores;
}

VecSimQueryResult_List BruteForceIndex::topKQuery(const void *queryBlob, size_t k,
                                                  VecSimQueryParams *queryParams) {

    VecSimQueryResult_List rl = {0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : NULL;

    this->last_mode = STANDARD_KNN;
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
        auto scores = computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            return rl;
        }
        for (size_t i = 0; i < scores.size(); i++) {
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
    rl.results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(rl.results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(rl.results[i], TopCandidates.top().first);
        TopCandidates.pop();
    }
    rl.code = VecSim_QueryResult_OK;
    return rl;
}

VecSimQueryResult_List BruteForceIndex::rangeQuery(const void *queryBlob, float radius,
                                                   VecSimQueryParams *queryParams) {
    auto rl = (VecSimQueryResult_List){0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    this->last_mode = RANGE_QUERY;

    float normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic when other types will be supported.
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(float));
        float_vector_normalize(normalized_blob, this->dim);
        queryBlob = normalized_blob;
    }

    // Compute scores in every block and save results that are within the range.
    rl.results =
        array_new<VecSimQueryResult>(10); // Use 10 as the initial capacity for the dynamic array.
    for (auto vectorBlock : this->vectorBlocks) {
        auto scores = computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            return rl;
        }
        for (size_t i = 0; i < scores.size(); i++) {
            if (scores[i] <= radius) {
                auto res = VecSimQueryResult{vectorBlock->getMember(i)->label, scores[i]};
                rl.results = array_append(rl.results, res);
            }
        }
    }
    rl.code = VecSim_QueryResult_OK;
    return rl;
}

VecSimIndexInfo BruteForceIndex::info() const {

    VecSimIndexInfo info;
    info.algo = VecSimAlgo_BF;
    info.bfInfo.dim = this->dim;
    info.bfInfo.type = this->vecType;
    info.bfInfo.metric = this->metric;
    info.bfInfo.indexSize = this->count;
    info.bfInfo.blockSize = this->vectorBlockSize;
    info.bfInfo.memory = this->allocator->getAllocationSize();
    info.bfInfo.last_mode = this->last_mode;
    return info;
}

VecSimInfoIterator *BruteForceIndex::infoIterator() {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed;
    size_t numberOfInfoFields = 8;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::ALGORITHM_STRING,
                                                .fieldType = INFOFIELD_STRING,
                                                .stringValue = VecSimAlgo_ToString(info.algo)});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::TYPE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .stringValue = VecSimType_ToString(info.bfInfo.type)});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.bfInfo.dim});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::METRIC_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .stringValue = VecSimMetric_ToString(info.bfInfo.metric)});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.bfInfo.indexSize});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.bfInfo.blockSize});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.bfInfo.memory});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .stringValue = VecSimSearchMode_ToString(info.bfInfo.last_mode)});

    return infoIterator;
}

VecSimBatchIterator *BruteForceIndex::newBatchIterator(const void *queryBlob,
                                                       VecSimQueryParams *queryParams) {
    // As this is the only supported type, we always allocate 4 bytes for every element in the
    // vector.
    assert(this->vecType == VecSimType_FLOAT32);
    auto *queryBlobCopy = this->allocator->allocate(sizeof(float) * this->dim);
    memcpy(queryBlobCopy, queryBlob, dim * sizeof(float));
    if (metric == VecSimMetric_Cosine) {
        float_vector_normalize((float *)queryBlobCopy, dim);
    }
    // Ownership of queryBlobCopy moves to BF_BatchIterator that will free it at the end.
    return new (this->allocator)
        BF_BatchIterator(queryBlobCopy, this, queryParams, this->allocator);
}

bool BruteForceIndex::preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) {
    // This heuristic is based on sklearn decision tree classifier (with 10 leaves nodes) -
    // see scripts/BF_batches_clf.py
    size_t index_size = this->indexSize();
    if (subsetSize > index_size) {
        throw std::runtime_error("internal error: subset size cannot be larger than index size");
    }
    size_t d = this->dim;
    float r = (index_size == 0) ? 0.0f : (float)(subsetSize) / (float)index_size;
    bool res;
    if (index_size <= 5500) {
        // node 1
        res = true;
    } else {
        // node 2
        if (d <= 300) {
            // node 3
            if (r <= 0.15) {
                // node 5
                res = true;
            } else {
                // node 6
                if (r <= 0.35) {
                    // node 9
                    if (d <= 75) {
                        // node 11
                        res = false;
                    } else {
                        // node 12
                        if (index_size <= 550000) {
                            // node 17
                            res = true;
                        } else {
                            // node 18
                            res = false;
                        }
                    }
                } else {
                    // node 10
                    res = false;
                }
            }
        } else {
            // node 4
            if (r <= 0.55) {
                // node 7
                res = true;
            } else {
                // node 8
                if (d <= 750) {
                    // node 13
                    res = false;
                } else {
                    // node 14
                    if (r <= 0.75) {
                        // node 15
                        res = true;
                    } else {
                        // node 16
                        res = false;
                    }
                }
            }
        }
    }
    // Set the mode - if this isn't the initial check, we switched mode form batches to ad-hoc.
    this->last_mode =
        res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
    return res;
}
