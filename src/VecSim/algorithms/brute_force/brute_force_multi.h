/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "brute_force.h"
#include "bfm_batch_iterator.h"
#include "VecSim/utils/updatable_heap.h"
#include "VecSim/utils/vec_utils.h"

template <typename DataType, typename DistType>
class BruteForceIndex_Multi : public BruteForceIndex<DataType, DistType> {
private:
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>> labelToIdsLookup;

public:
    BruteForceIndex_Multi(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
        : BruteForceIndex<DataType, DistType>(params, allocator), labelToIdsLookup(allocator) {}

    ~BruteForceIndex_Multi() {}

    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    int deleteVector(labelType labelType) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    inline size_t indexLabelCount() const override { return this->labelToIdsLookup.size(); }

    inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::unique_results_container(cap, this->allocator));
    }
#ifdef BUILD_TESTS
    void GetDataByLabel(labelType label, std::vector<std::vector<DataType>> &vectors_output) {

        auto ids = labelToIdsLookup.find(label);

        for (idType id : ids->second) {
            auto vec = std::vector<DataType>(this->dim);
            memcpy(vec.data(), this->getDataByInternalId(id), this->dim * sizeof(DataType));
            vectors_output.push_back(vec);
        }
    }
#endif
private:
    // inline definitions

    inline void setVectorId(labelType label, idType id) override;

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;

    inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() override {
        return new (this->allocator)
            vecsim_stl::updatable_max_heap<DistType, labelType>(this->allocator);
    }

    inline BF_BatchIterator<DataType, DistType> *
    newBatchIterator_Instance(void *queryBlob, VecSimQueryParams *queryParams) const override {
        return new (this->allocator)
            BFM_BatchIterator<DataType, DistType>(queryBlob, this, queryParams, this->allocator);
    }

#ifdef BUILD_TESTS
#include "VecSim/algorithms/brute_force/brute_force_multi_tests_friends.h"
#endif
};

/******************************* Implementation **********************************/

template <typename DataType, typename DistType>
int BruteForceIndex_Multi<DataType, DistType>::addVector(const void *vector_data, labelType label,
                                                         bool overwrite_allowed) {

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine.
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob, vector_data, this->dim * sizeof(DataType));
        normalizeVector(normalized_blob, this->dim);
        vector_data = normalized_blob;
    }

    this->appendVector(vector_data, label);
    return 1;
}

template <typename DataType, typename DistType>
int BruteForceIndex_Multi<DataType, DistType>::deleteVector(labelType label) {
    int ret = 0;

    // Find the id to delete.
    auto deleted_label_ids_pair = this->labelToIdsLookup.find(label);
    if (deleted_label_ids_pair == this->labelToIdsLookup.end()) {
        // Nothing to delete.
        return ret;
    }

    // Deletes all vectors under the given label.
    for (auto id_to_delete : deleted_label_ids_pair->second) {
        this->removeVector(id_to_delete);
        ret++;
    }

    // Remove the pair of the deleted vector.
    labelToIdsLookup.erase(label);
    return ret;
}

template <typename DataType, typename DistType>
double BruteForceIndex_Multi<DataType, DistType>::getDistanceFrom(labelType label,
                                                                  const void *vector_data) const {

    auto IDs = this->labelToIdsLookup.find(label);
    if (IDs == this->labelToIdsLookup.end()) {
        return INVALID_SCORE;
    }

    DistType dist = std::numeric_limits<DistType>::infinity();
    for (auto id : IDs->second) {
        DistType d = this->dist_func(this->getDataByInternalId(id), vector_data, this->dim);
        dist = (dist < d) ? dist : d;
    }

    return dist;
}

template <typename DataType, typename DistType>
void BruteForceIndex_Multi<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                                 idType old_id) {
    assert(labelToIdsLookup.find(label) != labelToIdsLookup.end());
    auto &ids = labelToIdsLookup.at(label);
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] == old_id) {
            ids[i] = new_id;
            return;
        }
    }
    assert(!"should have found the old id");
}

template <typename DataType, typename DistType>
void BruteForceIndex_Multi<DataType, DistType>::setVectorId(labelType label, idType id) {
    auto ids = labelToIdsLookup.find(label);
    if (ids != labelToIdsLookup.end()) {
        ids->second.push_back(id);
    } else {
        // Initial capacity is 1. We can consider increasing this value or having it as a
        // parameter.
        labelToIdsLookup.emplace(label, vecsim_stl::vector<idType>{1, id, this->allocator});
    }
}
