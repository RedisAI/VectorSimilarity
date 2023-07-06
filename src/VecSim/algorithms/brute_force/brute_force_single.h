/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "brute_force.h"
#include "bfs_batch_iterator.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/minmax_heap.h"

template <typename DataType, typename DistType>
class BruteForceIndex_Single : public BruteForceIndex<DataType, DistType> {

protected:
    vecsim_stl::unordered_map<labelType, idType> labelToIdLookup;

public:
    BruteForceIndex_Single(const BFParams *params,
                           const AbstractIndexInitParams &abstractInitParams);
    ~BruteForceIndex_Single();

    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override;
    int deleteVector(labelType label) override;
    int deleteVectorById(labelType label, idType id) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;

    inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::default_results_container(cap, this->allocator));
    }

    inline size_t indexLabelCount() const override { return this->count; }
    std::unordered_map<idType, std::pair<idType, labelType>>
    deleteVectorAndGetUpdatedIds(labelType label) override;

    // We call this when we KNOW that the label exists in the index.
    idType getIdOfLabel(labelType label) const { return labelToIdLookup.find(label)->second; }

#ifdef BUILD_TESTS
    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override {

        auto id = labelToIdLookup.at(label);

        auto vec = std::vector<DataType>(this->dim);
        memcpy(vec.data(), this->getDataByInternalId(id), this->dim * sizeof(DataType));
        vectors_output.push_back(vec);
    }
#endif
protected:
    // inline definitions

    inline void updateVector(idType id, const void *vector_data) {

        // Get the vector block
        DataBlock &vectorBlock = this->getVectorVectorBlock(id);
        size_t index = BruteForceIndex<DataType, DistType>::getVectorRelativeIndex(id);

        // Update vector data in the block.
        vectorBlock.updateElement(index, vector_data);
    }

    inline void setVectorId(labelType label, idType id) override {
        labelToIdLookup.emplace(label, id);
    }

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override {
        labelToIdLookup.at(label) = new_id;
    }

    inline void resizeLabelLookup(size_t new_max_elements) override {
        labelToIdLookup.reserve(new_max_elements);
    }

    inline bool isLabelExists(labelType label) override {
        return labelToIdLookup.find(label) != labelToIdLookup.end();
    }
    // Return a set of all labels that are stored in the index (helper for computing label count
    // without duplicates in tiered index). Caller should hold the flat buffer lock for read.
    inline vecsim_stl::set<labelType> getLabelsSet() const override {
        vecsim_stl::set<labelType> keys(this->allocator);
        for (auto &it : labelToIdLookup) {
            keys.insert(it.first);
        }
        return keys;
    };

    inline std::unique_ptr<vecsim_stl::abstract_min_max_heap<pair<DistType, labelType>>>
    getTopKCandidates(const void *queryBlob, size_t k, void *timeoutCtx) const override;

    inline BF_BatchIterator<DataType, DistType> *
    newBatchIterator_Instance(void *queryBlob, VecSimQueryParams *queryParams) const override {
        return new (this->allocator)
            BFS_BatchIterator<DataType, DistType>(queryBlob, this, queryParams, this->allocator);
    }

#ifdef BUILD_TESTS
#include "VecSim/algorithms/brute_force/brute_force_friend_tests.h"

#endif
};

/******************************* Implementation **********************************/

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::BruteForceIndex_Single(
    const BFParams *params, const AbstractIndexInitParams &abstractInitParams)
    : BruteForceIndex<DataType, DistType>(params, abstractInitParams),
      labelToIdLookup(this->allocator) {}

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::~BruteForceIndex_Single() {}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::addVector(const void *vector_data, labelType label,
                                                          void *auxiliaryCtx) {

    auto optionalID = this->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != this->labelToIdLookup.end()) {
        idType id = optionalID->second;
        updateVector(id, vector_data);
        return 0;
    }

    this->appendVector(vector_data, label);
    return 1;
}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::deleteVector(labelType label) {

    // Find the id to delete.
    auto deleted_label_id_pair = this->labelToIdLookup.find(label);
    if (deleted_label_id_pair == this->labelToIdLookup.end()) {
        // Nothing to delete.
        return 0;
    }

    // Get deleted vector id.
    idType id_to_delete = deleted_label_id_pair->second;

    // Remove the pair of the deleted vector.
    labelToIdLookup.erase(label);

    this->removeVector(id_to_delete);
    return 1;
}

template <typename DataType, typename DistType>
std::unordered_map<idType, std::pair<idType, labelType>>
BruteForceIndex_Single<DataType, DistType>::deleteVectorAndGetUpdatedIds(labelType label) {

    std::unordered_map<idType, std::pair<idType, labelType>> updated_ids;
    // Find the id to delete.
    auto deleted_label_id_pair = this->labelToIdLookup.find(label);
    if (deleted_label_id_pair == this->labelToIdLookup.end()) {
        // Nothing to delete.
        return updated_ids;
    }

    // Get deleted vector id.
    idType id_to_delete = deleted_label_id_pair->second;

    // Remove the pair of the deleted vector.
    labelToIdLookup.erase(label);
    labelType last_id_label = this->idToLabelMapping[this->count - 1];
    this->removeVector(id_to_delete); // this will decrease this->count and make the swap
    if (id_to_delete != this->count) {
        updated_ids[id_to_delete] = {this->count, last_id_label};
    }
    return updated_ids;
}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::deleteVectorById(labelType label, idType id) {
    return deleteVector(label);
}

template <typename DataType, typename DistType>
double BruteForceIndex_Single<DataType, DistType>::getDistanceFrom(labelType label,
                                                                   const void *vector_data) const {

    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = optionalId->second;

    return this->distFunc(this->getDataByInternalId(id), vector_data, this->dim);
}

template <typename DataType, typename DistType>
std::unique_ptr<vecsim_stl::abstract_min_max_heap<pair<DistType, labelType>>>
BruteForceIndex_Single<DataType, DistType>::getTopKCandidates(const void *queryBlob, size_t k,
                                                              void *timeoutCtx) const {

    auto *topCandidates = new (this->allocator)
        vecsim_stl::min_max_heap<pair<DistType, labelType>>(k, this->allocator);
    VecSimQueryResult_Code cur_block_code = VecSim_QueryResult_OK;

    DistType upperBound = std::numeric_limits<DistType>::lowest();
    // For every block, compute its vectors scores and update the Top candidates max heap
    idType cur_id = 0;
    for (auto &vectorBlock : this->vectorBlocks) {
        auto scores = this->computeBlockScores(vectorBlock, queryBlob, timeoutCtx, &cur_block_code);
        if (VecSim_OK != cur_block_code) {
            return nullptr;
        }
        for (size_t i = 0; i < scores.size(); i++) {
            if (topCandidates->size() < k) {
                // If we have less than k results, insert it.
                topCandidates->insert(std::make_pair(scores[i], this->getVectorLabel(cur_id)));
                upperBound = topCandidates->peek_max().first;
            } else if (scores[i] < upperBound) {
                // If we have result with a better score, insert it.
                topCandidates->exchange_max(
                    std::make_pair(scores[i], this->getVectorLabel(cur_id)));
                upperBound = topCandidates->peek_max().first;
            }
            ++cur_id;
        }
    }
    assert(cur_id == this->indexSize());
    return std::unique_ptr<vecsim_stl::abstract_min_max_heap<pair<DistType, labelType>>>(
        topCandidates);
}
