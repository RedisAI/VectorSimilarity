/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "hnsw.h"
#include "hnsw_single_batch_iterator.h"

template <typename DataType, typename DistType>
class HNSWIndex_Single : public HNSWIndex<DataType, DistType> {
private:
    // Index global state - this should be guarded by the indexDataGuard lock in
    // multithreaded scenario.
    vecsim_stl::unordered_map<labelType, idType> labelLookup;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_single_tests_friends.h"
#endif

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override { labelLookup[label] = id; }
    inline void resizeLabelLookup(size_t new_max_elements) override;
    inline vecsim_stl::set<labelType> getLabelsSet() const override;
    inline vecsim_stl::vector<idType> getElementIds(size_t label) override;
    inline double getDistanceFromInternal(labelType label, const void *vector_data) const;

public:
    HNSWIndex_Single(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                     const IndexComponents<DataType, DistType> &components,
                     size_t random_seed = 100)
        : HNSWIndex<DataType, DistType>(params, abstractInitParams, components, random_seed),
          labelLookup(this->allocator) {}
#ifdef BUILD_TESTS
    // Ctor to be used before loading a serialized index. Can be used from v2 and up.
    HNSWIndex_Single(std::ifstream &input, const HNSWParams *params,
                     const AbstractIndexInitParams &abstractInitParams,
                     const IndexComponents<DataType, DistType> &components,
                     Serializer::EncodingVersion version)
        : HNSWIndex<DataType, DistType>(input, params, abstractInitParams, components, version),
          labelLookup(this->maxElements, this->allocator) {}

    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override {

        auto id = labelLookup.at(label);

        auto vec = std::vector<DataType>(this->dim);
        // Only copy the vector data (dim * sizeof(DataType)), not any additional metadata like the
        // norm
        memcpy(vec.data(), this->getDataByInternalId(id), this->dim * sizeof(DataType));
        vectors_output.push_back(vec);
    }

    std::vector<std::vector<char>> getStoredVectorDataByLabel(labelType label) const override {
        std::vector<std::vector<char>> vectors_output;
        auto id = labelLookup.at(label);
        const char *data = this->getDataByInternalId(id);

        // Create a vector with the full data (including any metadata like norms)
        std::vector<char> vec(this->dataSize);
        memcpy(vec.data(), data, this->dataSize);
        vectors_output.push_back(std::move(vec));

        return vectors_output;
    }
#endif
    ~HNSWIndex_Single() = default;

    candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
    }
    std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::default_results_container(cap, this->allocator));
    }
    size_t indexLabelCount() const override;
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
size_t HNSWIndex_Single<DataType, DistType>::indexLabelCount() const {
    return labelLookup.size();
}

/**
 * helper functions
 */

// Return all the labels in the index - this should be used for computing the number of distinct
// labels in a tiered index, and caller should hold the index data guard.
template <typename DataType, typename DistType>
vecsim_stl::set<labelType> HNSWIndex_Single<DataType, DistType>::getLabelsSet() const {
    vecsim_stl::set<labelType> keys(this->allocator);
    for (auto &it : labelLookup) {
        keys.insert(it.first);
    }
    return keys;
}

template <typename DataType, typename DistType>
double
HNSWIndex_Single<DataType, DistType>::getDistanceFromInternal(labelType label,
                                                              const void *vector_data) const {

    auto it = labelLookup.find(label);
    if (it == labelLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = it->second;

    return this->calcDistance(vector_data, this->getDataByInternalId(id));
}

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                            idType old_id) {
    labelLookup[label] = new_id;
}

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::resizeLabelLookup(size_t new_max_elements) {
    labelLookup.reserve(new_max_elements);
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::deleteVector(const labelType label) {
    // Check that the label actually exists in the graph, and update the number of elements.
    if (labelLookup.find(label) == labelLookup.end()) {
        return 0;
    }
    idType element_internal_id = labelLookup[label];
    labelLookup.erase(label);
    this->removeVectorInPlace(element_internal_id);
    return 1;
}

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::addVector(const void *vector_data,
                                                    const labelType label) {
    // Checking if an element with the given label already exists.
    bool label_exists = labelLookup.find(label) != labelLookup.end();
    if (label_exists) {
        // Remove the vector in place if override allowed (in non-async scenario).
        deleteVector(label);
    }

    this->appendVector(vector_data, label);
    return label_exists ? 0 : 1;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Single<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                       VecSimQueryParams *queryParams) const {
    // force_copy == true.
    auto queryBlobCopy = this->preprocessQuery(queryBlob, true);

    // take ownership of the blob copy and pass it to the batch iterator.
    auto *queryBlobCopyPtr = queryBlobCopy.release();
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator) HNSWSingle_BatchIterator<DataType, DistType>(
        queryBlobCopyPtr, this, queryParams, this->allocator);
}

/**
 * Marks an element with the given label deleted, does NOT really change the current graph.
 * @param label
 */
template <typename DataType, typename DistType>
vecsim_stl::vector<idType> HNSWIndex_Single<DataType, DistType>::markDelete(labelType label) {
    std::unique_lock<std::shared_mutex> index_data_lock(this->indexDataGuard);
    auto internal_ids = this->getElementIds(label);
    if (!internal_ids.empty()) {
        assert(internal_ids.size() == 1); // expect to have only one id in index of type "single"
        this->markDeletedInternal(internal_ids[0]);
        labelLookup.erase(label);
    }
    return internal_ids;
}

template <typename DataType, typename DistType>
inline vecsim_stl::vector<idType>
HNSWIndex_Single<DataType, DistType>::getElementIds(size_t label) {
    vecsim_stl::vector<idType> ids(this->allocator);
    auto it = labelLookup.find(label);
    if (it == labelLookup.end()) {
        return ids;
    }
    ids.push_back(it->second);
    return ids;
}
