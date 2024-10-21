/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/containers/data_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/containers/vecsim_results_container.h"
#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/containers/data_blocks_container.h"
#include "VecSim/containers/raw_data_container_interface.h"
#include "VecSim/utils/vec_utils.h"

#include <cstring>
#include <memory>
#include <cassert>
#include <limits>
#include <ranges>
#include <sys/param.h>

using spaces::dist_func_t;

template <typename DataType, typename DistType>
class BruteForceIndex : public VecSimIndexAbstract<DataType, DistType> {
protected:
    vecsim_stl::vector<labelType> idToLabelMapping;
    RawDataContainer *vectors;
    idType count;

public:
    BruteForceIndex(const BFParams *params, const AbstractIndexInitParams &abstractInitParams,
                    const IndexComponents<DataType, DistType> &components);

    size_t indexSize() const override;
    size_t indexCapacity() const override;
    std::unique_ptr<RawDataContainer::Iterator> getVectorsIterator() const;
    DataType *getDataByInternalId(idType id) const { return (DataType *)vectors->getElement(id); }
    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override;
    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const override;
    VecSimIndexInfo info() const override;
    VecSimInfoIterator *infoIterator() const override;
    VecSimIndexBasicInfo basicInfo() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override;
    labelType getVectorLabel(idType id) const { return idToLabelMapping.at(id); }

    const RawDataContainer *getVectorsContainer() const { return vectors; }

    const labelType getLabelByInternalId(idType internal_id) const {
        return idToLabelMapping.at(internal_id);
    }
    // Remove a specific vector that is stored under a label from the index by its internal id.
    virtual int deleteVectorById(labelType label, idType id) = 0;
    // Remove a vector and return a map between internal ids and the original internal ids of the
    // vector that they hold as a result of the overall removals and swaps, along with its label.
    virtual std::unordered_map<idType, std::pair<idType, labelType>>
    deleteVectorAndGetUpdatedIds(labelType label) = 0;
    // Check if a certain label exists in the index.
    virtual bool isLabelExists(labelType label) = 0;
    // Return a set of all labels that are stored in the index (helper for computing label count
    // without duplicates in tiered index). Caller should hold the flat buffer lock for read.
    virtual vecsim_stl::set<labelType> getLabelsSet() const = 0;

    virtual ~BruteForceIndex() { delete vectors; }
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
    void fitMemory() override {
        idToLabelMapping.shrink_to_fit();
        resizeLabelLookup(idToLabelMapping.size());
    }
#endif

protected:
    // Private internal function that implements generic single vector insertion.
    virtual void appendVector(const void *vector_data, labelType label);

    // Private internal function that implements generic single vector deletion.
    virtual void removeVector(idType id);

    void growByBlock() {
        idToLabelMapping.resize(idToLabelMapping.size() + this->blockSize);
        idToLabelMapping.shrink_to_fit();
        resizeLabelLookup(idToLabelMapping.size());
    }

    void shrinkByBlock() {
        assert(indexCapacity() > 0); // should not be called when index is empty

        // remove a block size of labels.
        assert(idToLabelMapping.size() >= this->blockSize);
        idToLabelMapping.resize(idToLabelMapping.size() - this->blockSize);
        idToLabelMapping.shrink_to_fit();
        resizeLabelLookup(idToLabelMapping.size());
    }

    void setVectorLabel(idType id, labelType new_label) { idToLabelMapping.at(id) = new_label; }
    // inline priority queue getter that need to be implemented by derived class
    virtual vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() const = 0;

    // inline label to id setters that need to be implemented by derived class
    virtual std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const = 0;

    // inline label to id setters that need to be implemented by derived class
    virtual void replaceIdOfLabel(labelType label, idType new_id, idType old_id) = 0;
    virtual void setVectorId(labelType label, idType id) = 0;
    virtual void resizeLabelLookup(size_t new_max_elements) = 0;

    virtual VecSimBatchIterator *
    newBatchIterator_Instance(void *queryBlob, VecSimQueryParams *queryParams) const = 0;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/brute_force/brute_force_friend_tests.h"
#endif
};

/******************************* Implementation **********************************/

/******************** Ctor / Dtor **************/
template <typename DataType, typename DistType>
BruteForceIndex<DataType, DistType>::BruteForceIndex(
    const BFParams *params, const AbstractIndexInitParams &abstractInitParams,
    const IndexComponents<DataType, DistType> &components)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams, components),
      idToLabelMapping(this->allocator), count(0) {
    assert(VecSimType_sizeof(this->vecType) == sizeof(DataType));
    // Round up the initial capacity to the nearest multiple of the block size.
    size_t initialCapacity = RoundUpInitialCapacity(params->initialCapacity, this->blockSize);
    this->idToLabelMapping.resize(initialCapacity);
    vectors = new (this->allocator)
        DataBlocksContainer(this->blockSize, this->dataSize, this->allocator, this->alignment);
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
void BruteForceIndex<DataType, DistType>::appendVector(const void *vector_data, labelType label) {
    auto processed_blob = this->preprocessForStorage(vector_data);
    // Give the vector new id and increase count.
    idType id = this->count++;

    // Resize the index meta data structures if needed
    if (indexSize() > indexCapacity()) {
        growByBlock();
    }
    // add vector data to vector raw data container
    vectors->addElement(processed_blob.get(), id);

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

        // Put data of last vector inplace of the deleted vector.
        const char *last_vector_data = vectors->getElement(last_idx);
        vectors->updateElement(id_to_delete, last_vector_data);
    }
    vectors->removeElement(last_idx);

    // If we reached to a multiply of a block size, we can reduce meta data structures size.
    if (this->count % this->blockSize == 0) {
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

template <typename DataType, typename DistType>
std::unique_ptr<RawDataContainer::Iterator>
BruteForceIndex<DataType, DistType>::getVectorsIterator() const {
    return vectors->getIterator();
}

template <typename DataType, typename DistType>
VecSimQueryReply *
BruteForceIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) const {

    auto rep = new VecSimQueryReply(this->allocator);
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : NULL;
    this->lastMode = STANDARD_KNN;

    if (0 == k) {
        return rep;
    }

    auto processed_query_ptr = this->preprocessQuery(queryBlob);
    const void *processed_query = processed_query_ptr.get();
    DistType upperBound = std::numeric_limits<DistType>::lowest();
    vecsim_stl::abstract_priority_queue<DistType, labelType> *TopCandidates =
        getNewMaxPriorityQueue();

    // For vector, compute its scores and update the Top candidates max heap
    auto vectors_it = vectors->getIterator();
    idType curr_id = 0;
    while (auto *vector = vectors_it->next()) {
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            rep->code = VecSim_QueryReply_TimedOut;
            delete TopCandidates;
            return rep;
        }
        auto score = this->calcDistance(vector, processed_query);
        // If we have less than k or a better score, insert it.
        if (score < upperBound || TopCandidates->size() < k) {
            TopCandidates->emplace(score, getVectorLabel(curr_id));
            if (TopCandidates->size() > k) {
                // If we now have more than k results, pop the worst one.
                TopCandidates->pop();
            }
            upperBound = TopCandidates->top().first;
        }
        ++curr_id;
    }
    assert(curr_id == this->count);

    rep->results.resize(TopCandidates->size());
    for (auto &result : std::ranges::reverse_view(rep->results)) {
        std::tie(result.score, result.id) = TopCandidates->top();
        TopCandidates->pop();
    }
    delete TopCandidates;
    return rep;
}

template <typename DataType, typename DistType>
VecSimQueryReply *
BruteForceIndex<DataType, DistType>::rangeQuery(const void *queryBlob, double radius,
                                                VecSimQueryParams *queryParams) const {
    auto processed_query_ptr = this->preprocessQuery(queryBlob);
    auto rep = new VecSimQueryReply(this->allocator);
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    this->lastMode = RANGE_QUERY;

    // Compute scores in every block and save results that are within the range.
    auto res_container =
        getNewResultsContainer(10); // Use 10 as the initial capacity for the dynamic array.

    DistType radius_ = DistType(radius);
    auto vectors_it = vectors->getIterator();
    idType curr_id = 0;
    const void *processed_query = processed_query_ptr.get();
    while (vectors_it->hasNext()) {
        if (VECSIM_TIMEOUT(timeoutCtx)) {
            rep->code = VecSim_QueryReply_TimedOut;
            break;
        }
        auto score = this->calcDistance(vectors_it->next(), processed_query);
        if (score <= radius_) {
            res_container->emplace(getVectorLabel(curr_id), score);
        }
        ++curr_id;
    }
    // assert only if the loop finished iterating all the ids (we didn't get rep->code !=
    // VecSim_OK).
    assert((rep->code != VecSim_OK || curr_id == this->count));
    rep->results = res_container->get_results();
    return rep;
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
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields, this->allocator);

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
    auto *queryBlobCopy =
        this->allocator->allocate_aligned(this->dataSize, this->preprocessors->getAlignment());
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    this->preprocessQueryInPlace(queryBlobCopy);
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
