/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/utils/data_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vecsim_results_container.h"
#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/vec_utils.h"

#include <cstring>
#include <cmath>
#include <memory>
#include <queue>
#include <cassert>
#include <limits>

using spaces::dist_func_t;

template <typename DataType, typename DistType>
class BruteForceIndex : public VecSimIndexAbstract<DistType> {
protected:
    vecsim_stl::vector<labelType> idToLabelMapping;
    vecsim_stl::vector<DataBlock> vectorBlocks;
    idType count;

public:
    BruteForceIndex(const BFParams *params, const AbstractIndexInitParams &abstractInitParams);

    size_t indexSize() const override;
    size_t indexCapacity() const override;
    vecsim_stl::vector<DistType> computeBlockScores(const DataBlock &block, const void *queryBlob,
                                                    void *timeoutCtx,
                                                    VecSimQueryResult_Code *rc) const;
    inline DataType *getDataByInternalId(idType id) const {
        return (DataType *)vectorBlocks.at(id / this->blockSize).getElement(id % this->blockSize);
    }
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) const override;
    virtual VecSimQueryResult_List rangeQuery(const void *queryBlob, double radius,
                                              VecSimQueryParams *queryParams) const override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override;
    inline labelType getVectorLabel(idType id) const { return idToLabelMapping.at(id); }

    inline const vecsim_stl::vector<DataBlock> &getVectorBlocks() const { return vectorBlocks; }
    inline const labelType getLabelByInternalId(idType internal_id) const {
        return idToLabelMapping.at(internal_id);
    }
    // Remove a specific vector that is stored under a label from the index by its internal id.
    virtual int deleteVectorById(labelType label, idType id) = 0;
    // Remove a vector and return a map between internal ids and the original internal ids of the
    // vector that they hold as a result of the overall removals and swaps, along with its label.
    virtual std::unordered_map<idType, std::pair<idType, labelType>>
    deleteVectorAndGetUpdatedIds(labelType label) = 0;
    // Check if a certain label exists in the index.
    virtual inline bool isLabelExists(labelType label) = 0;
    // Return a set of all labels that are stored in the index (helper for computing label count
    // without duplicates in tiered index). Caller should hold the flat buffer lock for read.
    virtual inline vecsim_stl::set<labelType> getLabelsSet() const = 0;

    virtual ~BruteForceIndex() = default;
#ifdef BUILD_TESTS
    /**
     * @brief Used for testing - store vector(s) data associated with a given label. This function
     * copies the vector(s)' data buffer(s) and place it in the output vector
     *
     * @param label
     * @param vectors_output empty vector to be modified, should store the blob(s) associated with
     * the label.
     */
    virtual void getDataByLabel(labelType label,
                                std::vector<std::vector<DataType>> &vectors_output) const = 0;
#endif

protected:
    // Private internal function that implements generic single vector insertion.
    virtual void appendVector(const void *vector_data, labelType label);

    // Private internal function that implements generic single vector deletion.
    virtual void removeVector(idType id);

    inline void growByBlock() {
        assert(vectorBlocks.size() == 0 || vectorBlocks.back().getLength() == this->blockSize);
        vectorBlocks.emplace_back(this->blockSize, this->dataSize, this->allocator,
                                  this->alignment);
        idToLabelMapping.resize(idToLabelMapping.size() + this->blockSize);
        idToLabelMapping.shrink_to_fit();
        resizeLabelLookup(idToLabelMapping.size());
    }

    inline void shrinkByBlock() {
        assert(indexCapacity() > 0); // should not be called when index is empty

        // remove last block (should be empty)
        assert(vectorBlocks.size() > 0 && vectorBlocks.back().getLength() == 0);
        vectorBlocks.pop_back();

        // remove a block size of labels.
        assert(idToLabelMapping.size() >= this->blockSize);
        idToLabelMapping.resize(idToLabelMapping.size() - this->blockSize);
        idToLabelMapping.shrink_to_fit();
        resizeLabelLookup(idToLabelMapping.size());
    }

    inline DataBlock &getVectorVectorBlock(idType id) {
        return vectorBlocks.at(id / this->blockSize);
    }
    inline size_t getVectorRelativeIndex(idType id) const { return id % this->blockSize; }
    inline void setVectorLabel(idType id, labelType new_label) {
        idToLabelMapping.at(id) = new_label;
    }
    // inline priority queue getter that need to be implemented by derived class
    virtual inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() const = 0;

    // inline label to id setters that need to be implemented by derived class
    virtual inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const = 0;

    // inline label to id setters that need to be implemented by derived class
    virtual inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) = 0;
    virtual inline void setVectorId(labelType label, idType id) = 0;
    virtual inline void resizeLabelLookup(size_t new_max_elements) = 0;

    virtual inline VecSimBatchIterator *
    newBatchIterator_Instance(void *queryBlob, VecSimQueryParams *queryParams) const = 0;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/brute_force/brute_force_friend_tests.h"
#endif
};

/******************************* Implementation **********************************/

/******************** Ctor / Dtor **************/
template <typename DataType, typename DistType>
BruteForceIndex<DataType, DistType>::BruteForceIndex(
    const BFParams *params, const AbstractIndexInitParams &abstractInitParams)
    : VecSimIndexAbstract<DistType>(abstractInitParams), idToLabelMapping(this->allocator),
      vectorBlocks(this->allocator), count(0) {
    assert(VecSimType_sizeof(this->vecType) == sizeof(DataType));
    // Round up the initial capacity to the nearest multiple of the block size.
    size_t initialCapacity = RoundUpInitialCapacity(params->initialCapacity, this->blockSize);
    this->idToLabelMapping.resize(initialCapacity);
    this->vectorBlocks.reserve(initialCapacity / this->blockSize);
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
void BruteForceIndex<DataType, DistType>::appendVector(const void *vector_data, labelType label) {
    // Give the vector new id and increase count.
    idType id = this->count++;

    // Resize the index if needed.
    if (indexSize() > indexCapacity()) {
        growByBlock();
    } else if (id % this->blockSize == 0) {
        // If we we didn't reach the initial capacity but the last block is full, add a new block
        // only.
        this->vectorBlocks.emplace_back(this->blockSize, this->dataSize, this->allocator,
                                        this->alignment);
    }

    // Get the last vectors block to store the vector in.
    DataBlock &vectorBlock = this->vectorBlocks.back();
    assert(&vectorBlock == &getVectorVectorBlock(id));

    // add vector data to vectorBlock
    vectorBlock.addElement(vector_data);

    // add label to idToLabelMapping
    setVectorLabel(id, label);

    // add id to label:id map
    setVectorId(label, id);
}

template <typename DataType, typename DistType>
void BruteForceIndex<DataType, DistType>::removeVector(idType id_to_delete) {

    // Get last vector id and label
    idType last_idx = --this->count;
    labelType last_idx_label = getVectorLabel(last_idx);

    // Get last vector data.
    DataBlock &last_vector_block = vectorBlocks.back();
    assert(&last_vector_block == &getVectorVectorBlock(last_idx));

    void *last_vector_data = last_vector_block.removeAndFetchLastElement();

    // If we are *not* trying to remove the last vector, update mapping and move
    // the data of the last vector in the index in place of the deleted vector.
    if (id_to_delete != last_idx) {
        assert(id_to_delete < last_idx);
        // Update idToLabelMapping.
        // Put the label of the last_id in the deleted_id.
        setVectorLabel(id_to_delete, last_idx_label);

        // Update label2id mapping.
        // Update this id in label:id pair of last index.
        replaceIdOfLabel(last_idx_label, id_to_delete, last_idx);

        // Get the vectorBlock and the relative index of the deleted id.
        DataBlock &deleted_vectorBlock = getVectorVectorBlock(id_to_delete);
        size_t id_to_delete_rel_idx = getVectorRelativeIndex(id_to_delete);

        // Put data of last vector inplace of the deleted vector.
        deleted_vectorBlock.updateElement(id_to_delete_rel_idx, last_vector_data);
    }

    // If the last vector block is emtpy.
    if (last_vector_block.getLength() == 0) {
        shrinkByBlock();
    }
}

template <typename DataType, typename DistType>
size_t BruteForceIndex<DataType, DistType>::indexSize() const {
    return this->count;
}

template <typename DataType, typename DistType>
size_t BruteForceIndex<DataType, DistType>::indexCapacity() const {
    return this->idToLabelMapping.size();
}

// Compute the score for every vector in the block by using the given distance function.
template <typename DataType, typename DistType>
vecsim_stl::vector<DistType>
BruteForceIndex<DataType, DistType>::computeBlockScores(const DataBlock &block,
                                                        const void *queryBlob, void *timeoutCtx,
                                                        VecSimQueryResult_Code *rc) const {
    size_t len = block.getLength();
    vecsim_stl::vector<DistType> scores(len, this->allocator);
    for (size_t i = 0; i < len; i++) {
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            *rc = VecSim_QueryResult_TimedOut;
            return scores;
        }
        scores[i] = this->distFunc(block.getElement(i), queryBlob, this->dim);
    }
    *rc = VecSim_QueryResult_OK;
    return scores;
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
BruteForceIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) const {

    VecSimQueryResult_List rl = {0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : NULL;
    this->lastMode = STANDARD_KNN;

    if (0 == k) {
        rl.results = array_new<VecSimQueryResult>(0);
        return rl;
    }

    DistType upperBound = std::numeric_limits<DistType>::lowest();
    vecsim_stl::abstract_priority_queue<DistType, labelType> *TopCandidates =
        getNewMaxPriorityQueue();
    // For every block, compute its vectors scores and update the Top candidates max heap
    idType curr_id = 0;
    for (auto &vectorBlock : this->vectorBlocks) {
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
    assert(curr_id == this->count);

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

template <typename DataType, typename DistType>
VecSimQueryResult_List
BruteForceIndex<DataType, DistType>::rangeQuery(const void *queryBlob, double radius,
                                                VecSimQueryParams *queryParams) const {
    auto rl = (VecSimQueryResult_List){0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    this->lastMode = RANGE_QUERY;

    // Compute scores in every block and save results that are within the range.
    auto res_container =
        getNewResultsContainer(10); // Use 10 as the initial capacity for the dynamic array.

    DistType radius_ = DistType(radius);
    idType curr_id = 0;
    rl.code = VecSim_QueryResult_OK;
    for (auto &vectorBlock : this->vectorBlocks) {
        auto scores = computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            break;
        }
        for (size_t i = 0; i < scores.size(); i++) {
            if (scores[i] <= radius_) {
                res_container->emplace(getVectorLabel(curr_id), scores[i]);
            }
            ++curr_id;
        }
    }
    // assert only if the loop finished iterating all the ids (we didn't get rl.code != VecSim_OK).
    assert((rl.code != VecSim_OK || curr_id == this->count));
    rl.results = res_container->get_results();
    return rl;
}

template <typename DataType, typename DistType>
VecSimIndexInfo BruteForceIndex<DataType, DistType>::info() const {

    VecSimIndexInfo info;
    info.commonInfo = this->getCommonInfo();
    info.commonInfo.basicInfo.algo = VecSimAlgo_BF;

    return info;
}

template <typename DataType, typename DistType>
VecSimIndexBasicInfo BruteForceIndex<DataType, DistType>::basicInfo() const {

    VecSimIndexBasicInfo info = this->getBasicInfo();
    info.algo = VecSimAlgo_BF;
    info.isTiered = false;
    return info;
}

template <typename DataType, typename DistType>
VecSimInfoIterator *BruteForceIndex<DataType, DistType>::infoIterator() const {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 10;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::ALGORITHM_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .fieldValue = {FieldValue{
                             .stringValue = VecSimAlgo_ToString(info.commonInfo.basicInfo.algo)}}});
    this->addCommonInfoToIterator(infoIterator, info.commonInfo);
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
        .fieldType = INFOFIELD_UINT64,
        .fieldValue = {FieldValue{.uintegerValue = info.commonInfo.basicInfo.blockSize}}});
    return infoIterator;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
BruteForceIndex<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                      VecSimQueryParams *queryParams) const {
    auto *queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    // Ownership of queryBlobCopy moves to BF_BatchIterator that will free it at the end.
    return newBatchIterator_Instance(queryBlobCopy, queryParams);
}

template <typename DataType, typename DistType>
bool BruteForceIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                            bool initial_check) const {
    // This heuristic is based on sklearn decision tree classifier (with 10 leaves nodes) -
    // see scripts/BF_batches_clf.py
    size_t index_size = this->indexSize();
    // Referring to too large subset size as if it was the maximum possible size.
    subsetSize = std::min(subsetSize, index_size);

    size_t d = this->dim;
    float r = (index_size == 0) ? 0.0f : (float)(subsetSize) / (float)this->indexLabelCount();
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
    this->lastMode =
        res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
    return res;
}
