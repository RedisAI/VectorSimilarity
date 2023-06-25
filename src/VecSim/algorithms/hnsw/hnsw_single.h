/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
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

    template <bool Safe>
    inline double getDistanceFromInternal(labelType label, const void *vector_data) const;

public:
    HNSWIndex_Single(const HNSWParams *params, const AbstractIndexInitParams &abstractInitParams,
                     size_t random_seed = 100, size_t initial_pool_size = 1)
        : HNSWIndex<DataType, DistType>(params, abstractInitParams, random_seed, initial_pool_size),
          labelLookup(this->maxElements, this->allocator) {}
#ifdef BUILD_TESTS
    // Ctor to be used before loading a serialized index. Can be used from v2 and up.
    HNSWIndex_Single(std::ifstream &input, const HNSWParams *params,
                     const AbstractIndexInitParams &abstractInitParams,
                     Serializer::EncodingVersion version)
        : HNSWIndex<DataType, DistType>(input, params, abstractInitParams, version),
          labelLookup(this->maxElements, this->allocator) {}

    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override {

        auto id = labelLookup.at(label);

        auto vec = std::vector<DataType>(this->dim);
        memcpy(vec.data(), this->getDataByInternalId(id), this->dataSize);
        vectors_output.push_back(vec);
    }
#endif
    ~HNSWIndex_Single() {}

    inline candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
    }
    inline std::unique_ptr<vecsim_stl::abstract_results_container>
    getNewResultsContainer(size_t cap) const override {
        return std::unique_ptr<vecsim_stl::abstract_results_container>(
            new (this->allocator) vecsim_stl::default_results_container(cap, this->allocator));
    }

    inline size_t indexLabelCount() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override;

    int deleteVector(labelType label) override;
    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override;
    inline std::vector<idType> markDelete(labelType label) override;
    inline bool safeCheckIfLabelExistsInIndex(labelType label,
                                              bool also_done_processing = false) const override;

    double getDistanceFrom(labelType label, const void *vector_data) const override {
        return getDistanceFromInternal<false>(label, vector_data);
    }
    double safeGetDistanceFrom(labelType label, const void *vector_data) const override {
        return getDistanceFromInternal<true>(label, vector_data);
    }
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
inline vecsim_stl::set<labelType> HNSWIndex_Single<DataType, DistType>::getLabelsSet() const {
    vecsim_stl::set<labelType> keys(this->allocator);
    for (auto &it : labelLookup) {
        keys.insert(it.first);
    }
    return keys;
};

template <typename DataType, typename DistType>
template <bool Safe>
double
HNSWIndex_Single<DataType, DistType>::getDistanceFromInternal(labelType label,
                                                              const void *vector_data) const {
    if (Safe)
        this->indexDataGuard.lock_shared();

    auto it = labelLookup.find(label);
    if (it == labelLookup.end()) {
        if (Safe)
            this->indexDataGuard.unlock_shared();
        return INVALID_SCORE;
    }
    idType id = it->second;
    if (Safe)
        this->indexDataGuard.unlock_shared();

    return this->distFunc(vector_data, this->getDataByInternalId(id), this->dim);
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
int HNSWIndex_Single<DataType, DistType>::addVector(const void *vector_data, const labelType label,
                                                    void *auxiliaryCtx) {

    // Checking if an element with the given label already exists.
    bool label_exists = false;
    // Note that is it the caller responsibility to ensure that this label doesn't exist in the
    // index and increase the element count before calling this, if auxiliaryCtx is *not* NULL.
    if (auxiliaryCtx == nullptr) {
        if (labelLookup.find(label) != labelLookup.end()) {
            label_exists = true;
            // Remove the vector in place if override allowed (in non-async scenario).
            deleteVector(label);
        }
    }
    this->appendVector(vector_data, label, (AddVectorCtx *)auxiliaryCtx);
    // Return the delta in the index size due to the insertion.
    return label_exists ? 0 : 1;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Single<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                       VecSimQueryParams *queryParams) const {
    auto queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));

    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator) HNSWSingle_BatchIterator<DataType, DistType>(
        queryBlobCopy, this, queryParams, this->allocator);
}

/**
 * Marks an element with the given label deleted, does NOT really change the current graph.
 * @param label
 */
template <typename DataType, typename DistType>
std::vector<idType> HNSWIndex_Single<DataType, DistType>::markDelete(labelType label) {
    std::vector<idType> idsToDelete;
    std::unique_lock<std::shared_mutex> index_data_lock(this->indexDataGuard);
    auto search = labelLookup.find(label);
    if (search == labelLookup.end()) {
        return idsToDelete;
    }
    this->markDeletedInternal(search->second);
    idsToDelete.push_back(search->second);
    labelLookup.erase(search);
    return idsToDelete;
}

template <typename DataType, typename DistType>
inline bool HNSWIndex_Single<DataType, DistType>::safeCheckIfLabelExistsInIndex(
    labelType label, bool also_done_processing) const {
    std::unique_lock<std::shared_mutex> index_data_lock(this->indexDataGuard);
    auto it = labelLookup.find(label);
    bool exists = it != labelLookup.end();
    // If we want to make sure that the vector stored under the label was already indexed,
    // we go on and check that its associated internal id is no longer in process.
    if (exists && also_done_processing) {
        return !this->isInProcess(it->second);
    }
    return exists;
}
