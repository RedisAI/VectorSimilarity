#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/algorithms/brute_force/brute_force_factory.h"
#include "VecSim/algorithms/brute_force/bf_batch_iterator.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_struct.h"

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
    vecsim_stl::vector<VectorBlock *> vectorBlocks;
    idType count;

public:
    BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);

    virtual size_t indexSize() const override;
    vecsim_stl::vector<DistType> computeBlockScores(VectorBlock *block, const void *queryBlob,
                                                    void *timeoutCtx,
                                                    VecSimQueryResult_Code *rc) const;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, DistType radius,
                                      VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() const override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    inline labelType getVectorLabel(idType id) const { return idToLabelMapping.at(id); }

    inline vecsim_stl::vector<VectorBlock *> getVectorBlocks() const { return vectorBlocks; }
    virtual ~BruteForceIndex();

protected:
    // Private internal function that implements generic single vector insertion.
    virtual int appendVector(const void *vector_data, labelType label);

    // Private internal function that implements generic single vector deletion.
    virtual int removeVector(idType id);

    inline DataType *getDataByInternalId(idType id) const {
        return (DataType *)vectorBlocks.at(id / this->blockSize)->getVector(id % this->blockSize);
    }
    inline VectorBlock *getVectorVectorBlock(idType id) const {
        return vectorBlocks.at(id / this->blockSize);
    }
    inline size_t getVectorRelativeIndex(idType id) const { return id % this->blockSize; }
    inline void setVectorLabel(idType id, labelType new_label) {
        idToLabelMapping.at(id) = new_label;
    }

    // inline label to id setters that need to be implemented by derived class
    virtual inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) = 0;
    virtual inline void setVectorId(labelType label, idType id) = 0;

#ifdef BUILD_TESTS
    // Allow the following tests to access the index private members.
    friend class BruteForceTest_preferAdHocOptimization_Test;
    friend class BruteForceTest_test_dynamic_bf_info_iterator_Test;
    friend class BruteForceTest_resizeNAlignIndex_Test;
    friend class BruteForceTest_brute_force_vector_update_test_Test;
    friend class BruteForceTest_brute_force_reindexing_same_vector_Test;
    friend class BruteForceTest_test_delete_swap_block_Test;
    friend class BruteForceTest_brute_force_zero_minimal_capacity_Test;
    friend class BruteForceTest_resizeNAlignIndex_largeInitialCapacity_Test;
    friend class BruteForceTest_brute_force_empty_index_Test;
    friend class BM_VecSimBasics_DeleteVectorBF_Benchmark;
#endif
};

/******************************* Implementation **********************************/

/******************** Ctor / Dtor **************/
template <typename DataType, typename DistType>
BruteForceIndex<DataType, DistType>::BruteForceIndex(const BFParams *params,
                                                     std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric,
                                    params->blockSize, params->multi),
      idToLabelMapping(allocator), vectorBlocks(allocator), count(0) {
    assert(VecSimType_sizeof(this->vecType) == sizeof(DataType));
    this->idToLabelMapping.resize(params->initialCapacity);
}

template <typename DataType, typename DistType>
BruteForceIndex<DataType, DistType>::~BruteForceIndex() {
    for (auto &vectorBlock : this->vectorBlocks) {
        delete vectorBlock;
    }
}

/******************** Implementation **************/

template <typename DataType, typename DistType>
int BruteForceIndex<DataType, DistType>::appendVector(const void *vector_data, labelType label) {

    // Give the vector new id and increase count.
    idType id = count++;

    // Get vector block to store the vector in.

    // if vectorBlocks vector is empty or last_vector_block is full create a new block
    if (id % this->blockSize == 0) {
        size_t vector_bytes_count = this->dim * VecSimType_sizeof(this->vecType);
        VectorBlock *new_vectorBlock =
            new (this->allocator) VectorBlock(this->blockSize, vector_bytes_count, this->allocator);
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
        size_t last_block_vectors_count = id % this->blockSize;
        this->idToLabelMapping.resize(
            idToLabelMapping_size + this->blockSize - last_block_vectors_count, 0);
    }

    // add label to idToLabelMapping
    setVectorLabel(id, label);

    // add id to label:id map
    setVectorId(label, id);

    return true;
}

template <typename DataType, typename DistType>
int BruteForceIndex<DataType, DistType>::removeVector(idType id_to_delete) {

    // Get last vector id and label
    idType last_idx = --count;
    labelType last_idx_label = getVectorLabel(last_idx);

    // Get last vector data.
    VectorBlock *last_vector_block = vectorBlocks.back();
    assert(last_vector_block == getVectorVectorBlock(last_idx));

    char *last_vector_data = last_vector_block->removeAndFetchLastVector();

    // If we are *not* trying to remove the last vector, update mapping and move
    // the data of the last vector in the index in place of the deleted vector.
    if (id_to_delete != last_idx) {
        // Update id2labelmapping.
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

        // Resize and align the id2labelmapping.
        size_t id2label_size = idToLabelMapping.size();
        // If the new size is smaller by at least one block comparing to the id2labelmapping
        // align to be a multiplication of blocksize  and resize by one block.
        if (count + this->blockSize <= id2label_size) {
            size_t vector_to_align_count = id2label_size % this->blockSize;
            this->idToLabelMapping.resize(id2label_size - this->blockSize - vector_to_align_count);
        }
    }

    return true;
}

template <typename DataType, typename DistType>
size_t BruteForceIndex<DataType, DistType>::indexSize() const {
    return this->count;
}

// Compute the score for every vector in the block by using the given distance function.
template <typename DataType, typename DistType>
vecsim_stl::vector<DistType> BruteForceIndex<DataType, DistType>::computeBlockScores(
    VectorBlock *block, const void *queryBlob, void *timeoutCtx, VecSimQueryResult_Code *rc) const {
    size_t len = block->getLength();
    vecsim_stl::vector<DistType> scores(len, this->allocator);
    for (size_t i = 0; i < len; i++) {
        if (__builtin_expect(VecSimIndexAbstract<DistType>::timeoutCallback(timeoutCtx), 0)) {
            *rc = VecSim_QueryResult_TimedOut;
            return scores;
        }
        scores[i] = this->dist_func(block->getVector(i), queryBlob, this->dim);
    }
    *rc = VecSim_QueryResult_OK;
    return scores;
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
BruteForceIndex<DataType, DistType>::topKQuery(const void *queryBlob, size_t k,
                                               VecSimQueryParams *queryParams) {

    VecSimQueryResult_List rl = {0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : NULL;

    this->last_mode = STANDARD_KNN;
    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(DataType));
        float_vector_normalize(normalized_blob, this->dim);
        queryBlob = normalized_blob;
    }

    DistType upperBound = std::numeric_limits<DistType>::lowest();
    vecsim_stl::max_priority_queue<std::pair<DistType, labelType>> TopCandidates(this->allocator);
    // For every block, compute its vectors scores and update the Top candidates max heap
    idType curr_id = 0;
    for (auto vectorBlock : this->vectorBlocks) {
        auto scores = computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &rl.code);
        if (VecSim_OK != rl.code) {
            return rl;
        }
        for (size_t i = 0; i < scores.size(); i++) {
            // Always choose the current candidate if we have less than k.
            if (TopCandidates.size() < k) {
                TopCandidates.emplace(scores[i], getVectorLabel(curr_id));
                upperBound = TopCandidates.top().first;
            } else if (scores[i] < upperBound) {
                // Otherwise, try greedily to improve the top candidates with a vector that
                // has a better score than the one that has the worst score until now.
                TopCandidates.emplace(scores[i], getVectorLabel(curr_id));
                TopCandidates.pop();
                upperBound = TopCandidates.top().first;
            }
            ++curr_id;
        }
    }
    assert(curr_id == count);

    rl.results = array_new_len<VecSimQueryResult>(TopCandidates.size(), TopCandidates.size());
    for (int i = (int)TopCandidates.size() - 1; i >= 0; --i) {
        VecSimQueryResult_SetId(rl.results[i], TopCandidates.top().second);
        VecSimQueryResult_SetScore(rl.results[i], TopCandidates.top().first);
        TopCandidates.pop();
    }
    rl.code = VecSim_QueryResult_OK;
    return rl;
}

template <typename DataType, typename DistType>
VecSimQueryResult_List
BruteForceIndex<DataType, DistType>::rangeQuery(const void *queryBlob, DistType radius,
                                                VecSimQueryParams *queryParams) {
    auto rl = (VecSimQueryResult_List){0};
    void *timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
    this->last_mode = RANGE_QUERY;

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic when other types will be supported.
        memcpy(normalized_blob, queryBlob, this->dim * sizeof(DataType));
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

template <typename DataType, typename DistType>
VecSimIndexInfo BruteForceIndex<DataType, DistType>::info() const {

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

template <typename DataType, typename DistType>
VecSimInfoIterator *BruteForceIndex<DataType, DistType>::infoIterator() const {
    VecSimIndexInfo info = this->info();
    // For readability. Update this number when needed.
    size_t numberOfInfoFields = 8;
    VecSimInfoIterator *infoIterator = new VecSimInfoIterator(numberOfInfoFields);

    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimAlgo_ToString(info.algo)}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::TYPE_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimType_ToString(info.bfInfo.type)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::DIMENSION_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.dim}}});
    infoIterator->addInfoField(VecSim_InfoField{
        .fieldName = VecSimCommonStrings::METRIC_STRING,
        .fieldType = INFOFIELD_STRING,
        .fieldValue = {FieldValue{.stringValue = VecSimMetric_ToString(info.bfInfo.metric)}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::IS_MULTI_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.isMulti}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.indexSize}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::INDEX_LABEL_COUNT_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.indexLabelCount}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.blockSize}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::MEMORY_STRING,
                         .fieldType = INFOFIELD_UINT64,
                         .fieldValue = {FieldValue{.uintegerValue = info.bfInfo.memory}}});
    infoIterator->addInfoField(
        VecSim_InfoField{.fieldName = VecSimCommonStrings::SEARCH_MODE_STRING,
                         .fieldType = INFOFIELD_STRING,
                         .fieldValue = {FieldValue{
                             .stringValue = VecSimSearchMode_ToString(info.bfInfo.last_mode)}}});

    return infoIterator;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
BruteForceIndex<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                      VecSimQueryParams *queryParams) {
    assert(this->vecType == VecSimType_FLOAT32);
    auto *queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        float_vector_normalize((DataType *)queryBlobCopy, this->dim);
    }
    // Ownership of queryBlobCopy moves to BF_BatchIterator that will free it at the end.
    return new (this->allocator)
        BF_BatchIterator(queryBlobCopy, this, queryParams, this->allocator);
}

template <typename DataType, typename DistType>
bool BruteForceIndex<DataType, DistType>::preferAdHocSearch(size_t subsetSize, size_t k,
                                                            bool initial_check) {
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
