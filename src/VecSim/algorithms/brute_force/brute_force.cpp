
#include "brute_force.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/brute_force/brute_force_multi.h"

#include <memory>
#include <cstring>
#include <queue>
#include <cassert>
#include <cmath>

using namespace std;

/******************** Ctor / Dtor **************/
BruteForceIndex::BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract(allocator, params->dim, params->type, params->metric, params->blockSize,
                          params->multi),
      idToLabelMapping(allocator), vectorBlocks(allocator), count(0) {
    this->idToLabelMapping.resize(params->initialCapacity);
}

BruteForceIndex::~BruteForceIndex() {
    for (auto &vectorBlock : this->vectorBlocks) {
        delete vectorBlock;
    }
}

/******************** inheritance factory **************/

BruteForceIndex *BruteForceIndex::BruteForceIndex_New(const BFParams *params,
                                                      std::shared_ptr<VecSimAllocator> allocator) {
    if (params->multi)
        return new (allocator) BruteForceIndex_Multi(params, allocator);
    else
        return new (allocator) BruteForceIndex_Single(params, allocator);
}

/******************** Implementation **************/
size_t BruteForceIndex::estimateInitialSize(const BFParams *params) {
    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + sizeof(size_t);
    if (params->multi)
        est += sizeof(BruteForceIndex_Multi);
    else
        est += sizeof(BruteForceIndex_Single);

    // Parameters related part.

    if (params->initialCapacity) {
        est += params->initialCapacity * sizeof(decltype(idToLabelMapping)::value_type) +
               sizeof(size_t);
    }

    return est;
}

size_t BruteForceIndex::estimateElementMemory(const BFParams *params) {
    return params->dim * sizeof(float) + sizeof(idType);
}

int BruteForceIndex::appendVector(const void *vector_data, labelType label) {

    // Give the vector new id and increase count.
    idType id = count++;

    // Get vector block to store the vector in.

    // if vectorBlocks vector is empty or last_vector_block is full create a new block
    if (id % blockSize == 0) {
        VectorBlock *new_vectorBlock =
            new (this->allocator) VectorBlock(this->blockSize, this->dim, this->allocator);
        this->vectorBlocks.push_back(new_vectorBlock);
    }

    // get the last vectors block
    VectorBlock *vectorBlock = this->vectorBlocks.back();

    assert(vectorBlock == getVectorVectorBlock(id));

    // add vector data to vectorBlock
    vectorBlock->addVector(vector_data);

    // if idToLabelMapping is full,
    // resize and align idToLabelMapping by blockSize
    size_t idToLabelMapping_size = this->idToLabelMapping.size();

    if (id >= idToLabelMapping_size) {
        size_t last_block_vectors_count = id % blockSize;
        this->idToLabelMapping.resize(idToLabelMapping_size + blockSize - last_block_vectors_count,
                                      0);
    }

    // add label to idToLabelMapping
    setVectorLabel(id, label);

    // add id to label:id map
    setVectorId(label, id);

    return true;
}

int BruteForceIndex::removeVector(idType id_to_delete) {

    // Get last vector id and label
    idType last_idx = --count;
    labelType last_idx_label = getVectorLabel(last_idx);

    // Get last vector data.
    VectorBlock *last_vector_block = vectorBlocks.back();
    assert(last_vector_block == getVectorVectorBlock(last_idx));

    float *last_vector_data = last_vector_block->removeAndFetchLastVector();

    // If we are *not* trying to remove the last vector, update mapping and move
    // the data of the last vector in the index in place of the deleted vector.
    if (id_to_delete != last_idx) {
        // Update id2labelMapping.
        // Put the label of the last_id in the deleted_id.
        setVectorLabel(id_to_delete, last_idx_label);

        // Update label2id mapping.
        // Update this id in label:id pair of last index.
        replaceIdOfLabel(last_idx_label, id_to_delete, last_idx);

        // Get the vectorBlock and the relative index of the deleted id.
        VectorBlock *deleted_vectorBlock = getVectorVectorBlock(id_to_delete);
        size_t id_to_delete_rel_idx = getVectorRelativeIndex(id_to_delete);

        // Put data of last vector inplace of the deleted vector.
        deleted_vectorBlock->updateVector(id_to_delete_rel_idx, last_vector_data);
    }

    // If the last vector block is emtpy.
    if (last_vector_block->getLength() == 0) {
        delete last_vector_block;
        this->vectorBlocks.pop_back();

        // Resize and align the id2labelMapping.
        size_t id2label_size = idToLabelMapping.size();
        // If the new size is smaller by at least one block comparing to the id2labelMapping
        // align to be a multiplication of blocksize  and resize by one block.
        if (count + blockSize <= id2label_size) {
            size_t vector_to_align_count = id2label_size % blockSize;
            this->idToLabelMapping.resize(id2label_size - blockSize - vector_to_align_count);
        }
    }

    return true;
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
        if (__builtin_expect(VecSimIndexAbstract::timeoutCallback(timeoutCtx), 0)) {
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
    float normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(float));
        float_vector_normalize(normalized_blob, this->dim);
        queryBlob = normalized_blob;
    }

    float upperBound = std::numeric_limits<float>::lowest();
    vecsim_stl::abstract_priority_queue<float, labelType> *TopCandidates = getNewPriorityQueue();
    // For every block, compute its vectors scores and update the Top candidates max heap
    idType curr_id = 0;
    for (auto vectorBlock : this->vectorBlocks) {
        auto scores = computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            delete TopCandidates;
            return rl;
        }
        for (size_t i = 0; i < scores.size(); i++) {
            // If we have less than k or a better score, insert it.
            if (scores[i] < upperBound || TopCandidates->size() < k) {
                TopCandidates->emplace(scores[i], getVectorLabel(curr_id));
                if (TopCandidates->size() > k) {
                    // If we now have more than k results, pop the worst one.
                    TopCandidates->pop();
                }
                upperBound = TopCandidates->top().first;
            }
            ++curr_id;
        }
    }
    assert(curr_id == count);

    rl.results = array_new_len<VecSimQueryResult>(TopCandidates->size(), TopCandidates->size());
    for (int i = (int)TopCandidates->size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(rl.results[i], TopCandidates->top().second);
        VecSimQueryResult_SetScore(rl.results[i], TopCandidates->top().first);
        TopCandidates->pop();
    }
    delete TopCandidates;
    rl.code = VecSim_QueryResult_OK;
    return rl;
}

VecSimQueryResult_List BruteForceIndex::rangeQuery(const void *queryBlob, float radius,
                                                   VecSimQueryParams *queryParams) {
    auto rl = (VecSimQueryResult_List){0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    this->last_mode = RANGE_QUERY;

    float normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic when other types will be supported.
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(float));
        float_vector_normalize(normalized_blob, this->dim);
        queryBlob = normalized_blob;
    }

    // Compute scores in every block and save results that are within the range.
    rl.results =
        array_new<VecSimQueryResult>(10); // Use 10 as the initial capacity for the dynamic array.

    idType curr_id = 0;
    for (auto vectorBlock : this->vectorBlocks) {
        auto scores = computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            return rl;
        }
        for (size_t i = 0; i < scores.size(); i++) {
            if (scores[i] <= radius) {
                auto res = VecSimQueryResult{getVectorLabel(curr_id), scores[i]};
                rl.results = array_append(rl.results, res);
            }
            ++curr_id;
        }
    }
    assert(curr_id == count);
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
    info.bfInfo.indexLabelCount = this->indexLabelCount();
    info.bfInfo.blockSize = this->blockSize;
    info.bfInfo.memory = this->allocator->getAllocationSize();
    info.bfInfo.isMulti = this->isMulti;
    info.bfInfo.last_mode = this->last_mode;
    return info;
}

VecSimInfoIterator *BruteForceIndex::infoIterator() const {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
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
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.bfInfo.isMulti});
    infoIterator->addInfoField(VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                                                .fieldType = INFOFIELD_UINT64,
                                                .uintegerValue = info.bfInfo.indexSize});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .uintegerValue = info.bfInfo.indexLabelCount});
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
    return newBatchIterator_Instance(queryBlobCopy, queryParams);
}

bool BruteForceIndex::preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) {
    // This heuristic is based on sklearn decision tree classifier (with 10 leaves nodes) -
    // see scripts/BF_batches_clf.py
    size_t index_size = this->indexLabelCount(); // Index's unique label size
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
