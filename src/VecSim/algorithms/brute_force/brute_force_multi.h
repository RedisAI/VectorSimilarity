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
    BruteForceIndex_Multi(const BFParams *params, const AbstractIndexInitParams &abstractInitParams)
        : BruteForceIndex<DataType, DistType>(params, abstractInitParams),
          labelToIdsLookup(this->allocator) {}

    ~BruteForceIndex_Multi() {}

    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override;
    int deleteVector(labelType labelType) override;
    int deleteVectorById(labelType label, idType id) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    inline size_t indexLabelCount() const override { return this->labelToIdsLookup.size(); }

    inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::unique_results_container(cap, this->allocator));
    }
    std::unordered_map<idType, std::pair<idType, labelType>>
    deleteVectorAndGetUpdatedIds(labelType label) override;
#ifdef BUILD_TESTS
    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override {

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

    inline bool isLabelExists(labelType label) override {
        return labelToIdsLookup.find(label) != labelToIdsLookup.end();
    }
    // Return a set of all labels that are stored in the index (helper for computing label count
    // without duplicates in tiered index). Caller should hold the flat buffer lock for read.
    inline vecsim_stl::set<labelType> getLabelsSet() const override {
        vecsim_stl::set<labelType> keys(this->allocator);
        for (auto &it : labelToIdsLookup) {
            keys.insert(it.first);
        }
        return keys;
    };

    inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() const override {
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
                                                         void *auxiliaryCtx) {
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
std::unordered_map<idType, std::pair<idType, labelType>>
BruteForceIndex_Multi<DataType, DistType>::deleteVectorAndGetUpdatedIds(labelType label) {
    // Hold a mapping from ids that are removed and changed to the original ids that were swapped
    // into it. For example, if we have ids 0, 1, 2, 3, 4 and are about to remove ids 1, 3, 4, we
    // should get the following scenario: {1->4} => {1->4} => {1->2}.
    // Explanation: first we delete 1 and swap it with 4. Then, we remove 3 and have no swap since 3
    // is the last id. Lastly, we delete the original 4 which is now in id 1, and swap it with 2.
    // Eventually, in id 1 we should have the original vector whose id was 2.
    std::unordered_map<idType, std::pair<idType, labelType>> updated_ids;

    // Find the id to delete.
    auto deleted_label_ids_pair = this->labelToIdsLookup.find(label);
    if (deleted_label_ids_pair == this->labelToIdsLookup.end()) {
        // Nothing to delete.
        return updated_ids;
    }

    // Deletes all vectors under the given label.
    for (size_t i = 0; i < deleted_label_ids_pair->second.size(); i++) {
        idType cur_id_to_delete = deleted_label_ids_pair->second[i];
        // The removal take into consideration the current internal id to remove, even if it is not
        // the original id, and it has swapped into this id after previous swap of another id that
        // belongs to this label.
        labelType last_id_label = this->idToLabelMapping[this->count - 1];
        this->removeVector(cur_id_to_delete);
        // If cur_id_to_delete exists in the map, remove it as it is no longer valid, whether it
        // will get a new value due to a swap, or it is the last element in the index.
        updated_ids.erase(cur_id_to_delete);
        // If a swap was made, update who was the original id that now resides in cur_id_to_delete.
        if (cur_id_to_delete != this->count) {
            if (updated_ids.find(this->count) != updated_ids.end()) {
                updated_ids[cur_id_to_delete] = updated_ids[this->count];
                updated_ids.erase(this->count);
            } else {
                // Otherwise, the last id now resides where the deleted id was.
                updated_ids[cur_id_to_delete] = {this->count, last_id_label};
            }
        }
    }
    // Remove the pair of the deleted vector.
    labelToIdsLookup.erase(label);
    return updated_ids;
}

template <typename DataType, typename DistType>
int BruteForceIndex_Multi<DataType, DistType>::deleteVectorById(labelType label, idType id) {
    // Find the id to delete.
    auto deleted_label_ids_pair = this->labelToIdsLookup.find(label);
    if (deleted_label_ids_pair == this->labelToIdsLookup.end()) {
        // Nothing to delete.
        return 0;
    }

    // Delete the specific vector id which is under the given label.
    auto &ids = deleted_label_ids_pair->second;
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] == id) {
            this->removeVector(id);
            ids.erase(ids.begin() + i);
            if (ids.empty()) {
                labelToIdsLookup.erase(label);
            }
            return 1;
        }
    }
    assert(false && "id to delete was not found under the given label");
    return 0;
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
    // *Non-trivial code here* - in every iteration we replace the internal id of the previous last
    // id that has been swapped with the deleted id. Note that if the old and the new replaced ids
    // both belong to the same label, then we are going to delete the new id later on as well, since
    // we are currently iterating on this exact array of ids in 'deleteVector'. Hence, the relevant
    // part of the vector that should be updated is the "tail" that comes after the position of
    // old_id, while the "head" may contain old occurrences of old_id that are irrelevant for the
    // future deletions. Therefore, we iterate from end to beginning. For example, assuming we are
    // deleting a label that contains the only 3 ids that exist in the index. Hence, we would
    // expect the following scenario w.r.t. the ids array:
    // [|1, 0, 2] -> [1, |0, 1] -> [1, 0, |0] (where | marks the current position)
    auto &ids = labelToIdsLookup.at(label);
    for (int i = ids.size() - 1; i >= 0; i--) {
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
