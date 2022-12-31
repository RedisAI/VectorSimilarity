/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "brute_force.h"
#include "bfs_batch_iterator.h"
#include "VecSim/utils/vec_utils.h"

template <typename DataType, typename DistType>
class BruteForceIndex_Single : public BruteForceIndex<DataType, DistType> {

protected:
    vecsim_stl::unordered_map<labelType, idType> labelToIdLookup;

public:
    BruteForceIndex_Single(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    ~BruteForceIndex_Single();

    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    int deleteVector(labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;

    inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::default_results_container(cap, this->allocator));
    }

    inline size_t indexLabelCount() const override { return this->count; }
#ifdef BUILD_TESTS
    void GetDataByLabel(labelType label, std::vector<std::vector<DataType>> &vectors_output) {

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
        VectorBlock *vectorBlock = this->getVectorVectorBlock(id);
        size_t index = BruteForceIndex<DataType, DistType>::getVectorRelativeIndex(id);

        // Update vector data in the block.
        vectorBlock->updateVector(index, vector_data);
    }

    inline void setVectorId(labelType label, idType id) override {
        labelToIdLookup.emplace(label, id);
    }

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override {
        labelToIdLookup.at(label) = new_id;
    }

    inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
    }

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
    const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : BruteForceIndex<DataType, DistType>(params, allocator), labelToIdLookup(allocator) {}

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::~BruteForceIndex_Single() {}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::addVector(const void *vector_data, labelType label,
                                                          bool overwrite_allowed) {

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob, vector_data, this->dim * sizeof(DataType));
        normalizeVector(normalized_blob, this->dim);
        vector_data = normalized_blob;
    }

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
double BruteForceIndex_Single<DataType, DistType>::getDistanceFrom(labelType label,
                                                                   const void *vector_data) const {

    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = optionalId->second;

    return this->dist_func(this->getDataByInternalId(id), vector_data, this->dim);
}
