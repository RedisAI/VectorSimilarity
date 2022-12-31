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
    vecsim_stl::unordered_map<labelType, idType> label_lookup_;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_single_tests_friends.h"
#endif

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override { label_lookup_[label] = id; }
    inline void resizeLabelLookup(size_t new_max_elements) override;

public:
    HNSWIndex_Single(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
                     size_t random_seed = 100, size_t initial_pool_size = 1)
        : HNSWIndex<DataType, DistType>(params, allocator, random_seed, initial_pool_size),
          label_lookup_(this->max_elements_, allocator) {}
#ifdef BUILD_TESTS
    // Ctor to be used before loading a serialized index. Can be used from v2 and up.
    HNSWIndex_Single(std::ifstream &input, const HNSWParams *params,
                     std::shared_ptr<VecSimAllocator> allocator,
                     Serializer::EncodingVersion version)
        : HNSWIndex<DataType, DistType>(input, params, allocator, version),
          label_lookup_(this->max_elements_, allocator) {}

    void GetDataByLabel(labelType label, std::vector<std::vector<DataType>> &vectors_output) {

        auto id = label_lookup_.at(label);

        auto vec = std::vector<DataType>(this->dim);
        memcpy(vec.data(), this->getDataByInternalId(id), this->data_size_);
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
    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    inline std::vector<idType> markDelete(labelType label) override;
};

/**
 * getters and setters of index data
 */

template <typename DataType, typename DistType>
size_t HNSWIndex_Single<DataType, DistType>::indexLabelCount() const {
    return label_lookup_.size();
}

template <typename DataType, typename DistType>
double HNSWIndex_Single<DataType, DistType>::getDistanceFrom(labelType label,
                                                             const void *vector_data) const {
    auto id = label_lookup_.find(label);
    if (id == label_lookup_.end() || this->isMarkedDeleted(id->second)) {
        return INVALID_SCORE;
    }
    return this->dist_func(vector_data, this->getDataByInternalId(id->second), this->dim);
}

/**
 * helper functions
 */

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                            idType old_id) {
    label_lookup_[label] = new_id;
}

template <typename DataType, typename DistType>
void HNSWIndex_Single<DataType, DistType>::resizeLabelLookup(size_t new_max_elements) {
    label_lookup_.reserve(new_max_elements);
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::deleteVector(const labelType label) {
    // Check that the label actually exists in the graph, and update the number of elements.
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return 0;
    }
    idType element_internal_id = label_lookup_[label];
    label_lookup_.erase(label);
    this->removeVector(element_internal_id);
    return 1;
}

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::addVector(const void *vector_data, const labelType label,
                                                    bool overwrite_allowed) {

    // Checking if an element with the given label already exists.
    bool label_exists = false;
    if (label_lookup_.find(label) != label_lookup_.end()) {
        label_exists = true;
        if (overwrite_allowed) {
            // Remove the vector in place if override allowed (in non-async scenario)
            deleteVector(label);
        } else {
            // If override is not allowed, we don't do anything.
            return -1;
        }
    }

    this->appendVector(vector_data, label);
    // Return the delta in the index size due to the insertion.
    return label_exists ? 0 : 1;
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Single<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                       VecSimQueryParams *queryParams) const {
    auto queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        normalizeVector((DataType *)queryBlobCopy, this->dim);
    }
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
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        return idsToDelete;
    }
    this->markDeletedInternal(search->second);
    idsToDelete.push_back(search->second);
    label_lookup_.erase(search);
    return idsToDelete;
}
