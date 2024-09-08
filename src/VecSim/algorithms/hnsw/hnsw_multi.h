/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "hnsw.h"
#include "hnsw_multi_batch_iterator.h"
#include "VecSim/utils/updatable_heap.h"

template <typename DataType, typename DistType>
class HNSWIndex_Multi : public HNSWIndex<DataType, DistType> {
private:
    // Index global state - this should be guarded by the indexDataGuard lock in
    // multithreaded scenario.
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>> labelLookup;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_multi_tests_friends.h"
#endif

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override {
        // Checking if an element with the given label already exists.
        // if not, add an empty vector under the new label.
        if (labelLookup.find(label) == labelLookup.end()) {
            labelLookup.emplace(label, vecsim_stl::vector<idType>{this->allocator});
        }
        labelLookup.at(label).push_back(id);
    }
    inline vecsim_stl::vector<idType> getElementIds(size_t label) override {
        auto it = labelLookup.find(label);
        if (it == labelLookup.end()) {
            return vecsim_stl::vector<idType>{this->allocator}; // return an empty collection
        }
        return it->second;
    }
    inline void resizeLabelLookup(size_t new_max_elements) override;

    // Return all the labels in the index - this should be used for computing the number of distinct
    // labels in a tiered index, and caller should hold the index data guard.
    inline vecsim_stl::set<labelType> getLabelsSet() const override {
        vecsim_stl::set<labelType> keys(this->allocator);
        for (auto &it : labelLookup) {
            keys.insert(it.first);
        }
        return keys;
    };

    inline double getDistanceFromInternal(labelType label, const void *vector_data) const;

public:
    HNSWIndex_Multi(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                    IndexComputerAbstract<DistType> *indexComputer, size_t random_seed = 100,
                    size_t initial_pool_size = 1)
        : HNSWIndex<DataType, DistType>(params, abstractInitParams, indexComputer, random_seed,
                                        initial_pool_size),
          labelLookup(this->maxElements, this->allocator) {}
#ifdef BUILD_TESTS
    // Ctor to be used before loading a serialized index. Can be used from v2 and up.
    HNSWIndex_Multi(std::ifstream &input, const HNSWParams *params,
                    const AbstractIndexInitParams &abstractInitParams,
                    Serializer::EncodingVersion version,
                    IndexComputerAbstract<DistType> *indexComputer)
        : HNSWIndex<DataType, DistType>(input, params, abstractInitParams, version, indexComputer),
          labelLookup(this->maxElements, this->allocator) {}

    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override {

        auto ids = labelLookup.find(label);

        for (idType id : ids->second) {
            auto vec = std::vector<DataType>(this->dim);
            memcpy(vec.data(), this->getDataByInternalId(id), this->dataSize);
            vectors_output.push_back(vec);
        }
    }
#endif
    ~HNSWIndex_Multi() = default;

    inline candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const override {
        return new (this->allocator)
            vecsim_stl::updatable_max_heap<DistType, labelType>(this->allocator);
    }
    inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::unique_results_container(cap, this->allocator));
    }

    inline size_t indexLabelCount() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override;

    int deleteVector(labelType label) override;
    int addVector(const void *vector_data, labelType label) override;
    vecsim_stl::vector<idType> markDelete(labelType label) override;
    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        return getDistanceFromInternal(label, vector_data);
    }
    int removeLabel(labelType label) override { return labelLookup.erase(label); }
};

/**
 * getters and setters of index data
 */

template <typename DataType, typename DistType>
size_t HNSWIndex_Multi<DataType, DistType>::indexLabelCount() const {
    return labelLookup.size();
}

/**
 * helper functions
 */

template <typename DataType, typename DistType>
double HNSWIndex_Multi<DataType, DistType>::getDistanceFromInternal(labelType label,
                                                                    const void *vector_data) const {
    DistType dist = INVALID_SCORE;

    // Check if the label exists in the index, return invalid score if not.
    auto it = this->labelLookup.find(label);
    if (it == this->labelLookup.end()) {
        return dist;
    }

    // Get the vector of ids associated with the label.
    // Get a copy if `Safe` is true, otherwise get a reference.
    auto &IDs = it->second;

    // Iterate over the ids and find the minimum distance.
    for (auto id : IDs) {
        DistType d = this->indexComputer->calcDistance(this->getDataByInternalId(id), vector_data,
                                                       this->dim);
        dist = std::fmin(dist, d);
    }

    return dist;
}

template <typename DataType, typename DistType>
void HNSWIndex_Multi<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                           idType old_id) {
    assert(labelLookup.find(label) != labelLookup.end());
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
    auto &ids = labelLookup.at(label);
    for (int i = ids.size() - 1; i >= 0; i--) {
        if (ids[i] == old_id) {
            ids[i] = new_id;
            return;
        }
    }
    assert(!"should have found the old id");
}

template <typename DataType, typename DistType>
void HNSWIndex_Multi<DataType, DistType>::resizeLabelLookup(size_t new_max_elements) {
    labelLookup.reserve(new_max_elements);
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
int HNSWIndex_Multi<DataType, DistType>::deleteVector(const labelType label) {
    int ret = 0;
    // check that the label actually exists in the graph, and update the number of elements.
    auto ids_it = labelLookup.find(label);
    if (ids_it == labelLookup.end()) {
        return ret;
    }
    for (auto &ids = ids_it->second; idType id : ids) {
        this->removeVectorInPlace(id);
        ret++;
    }
    labelLookup.erase(label);
    return ret;
}

template <typename DataType, typename DistType>
int HNSWIndex_Multi<DataType, DistType>::addVector(const void *vector_data, const labelType label) {

    this->appendVector(vector_data, label);
    return 1; // We always add the vector, no overrides in multi.
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Multi<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                      VecSimQueryParams *queryParams) const {
    auto queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator) HNSWMulti_BatchIterator<DataType, DistType>(
        queryBlobCopy, this, queryParams, this->allocator);
}

/**
 * Marks an element with the given label deleted, does NOT really change the current graph.
 * @param label
 */
template <typename DataType, typename DistType>
vecsim_stl::vector<idType> HNSWIndex_Multi<DataType, DistType>::markDelete(labelType label) {
    std::unique_lock<std::shared_mutex> index_data_lock(this->indexDataGuard);

    auto ids = this->getElementIds(label);
    for (idType id : ids) {
        this->markDeletedInternal(id);
    }
    labelLookup.erase(label);
    return ids;
}
