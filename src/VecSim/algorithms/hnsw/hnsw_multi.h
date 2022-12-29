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
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>> label_lookup_;

#ifdef BUILD_TESTS
#include "VecSim/algorithms/hnsw/hnsw_multi_tests_friends.h"
#endif

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override {
        // Checking if an element with the given label already exists.
        // if not, add an empty vector under the new label.
        if (label_lookup_.find(label) == label_lookup_.end()) {
            label_lookup_.emplace(label, vecsim_stl::vector<idType>{this->allocator});
        }
        label_lookup_.at(label).push_back(id);
    }
    inline void resizeLabelLookup(size_t new_max_elements) override;

public:
    HNSWIndex_Multi(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
                    size_t random_seed = 100, size_t initial_pool_size = 1)
        : HNSWIndex<DataType, DistType>(params, allocator, random_seed, initial_pool_size),
          label_lookup_(this->max_elements_, allocator) {}
#ifdef BUILD_TESTS
    // Ctor to be used before loading a serialized index. Can be used from v2 and up.
    HNSWIndex_Multi(std::ifstream &input, const HNSWParams *params,
                    std::shared_ptr<VecSimAllocator> allocator, Serializer::EncodingVersion version)
        : HNSWIndex<DataType, DistType>(input, params, allocator, version),
          label_lookup_(this->max_elements_, allocator) {}

    void GetDataByLabel(labelType label, std::vector<std::vector<DataType>> &vectors_output) {

        auto ids = label_lookup_.find(label);

        for (idType id : ids->second) {
            auto vec = std::vector<DataType>(this->dim);
            memcpy(vec.data(), this->getDataByInternalId(id), this->data_size_);
            vectors_output.push_back(vec);
        }
    }
#endif
    ~HNSWIndex_Multi() {}

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
    int addVector(const void *vector_data, labelType label, bool overwrite_allowed = true) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    inline std::vector<idType> markDelete(labelType label) override;
};

/**
 * getters and setters of index data
 */

template <typename DataType, typename DistType>
size_t HNSWIndex_Multi<DataType, DistType>::indexLabelCount() const {
    return label_lookup_.size();
}

template <typename DataType, typename DistType>
double HNSWIndex_Multi<DataType, DistType>::getDistanceFrom(labelType label,
                                                            const void *vector_data) const {

    auto IDs = this->label_lookup_.find(label);
    if (IDs == this->label_lookup_.end()) {
        return INVALID_SCORE;
    }

    DistType dist = std::numeric_limits<DistType>::infinity();
    for (auto id : IDs->second) {
        if (!this->isMarkedDeleted(id)) {
            DistType d = this->dist_func(this->getDataByInternalId(id), vector_data, this->dim);
            dist = (dist < d) ? dist : d;
        }
    }

    return dist;
}

/**
 * helper functions
 */

template <typename DataType, typename DistType>
void HNSWIndex_Multi<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                           idType old_id) {
    assert(label_lookup_.find(label) != label_lookup_.end());
    auto &ids = label_lookup_.at(label);
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] == old_id) {
            ids[i] = new_id;
            return;
        }
    }
    assert(!"should have found the old id");
}

template <typename DataType, typename DistType>
void HNSWIndex_Multi<DataType, DistType>::resizeLabelLookup(size_t new_max_elements) {
    label_lookup_.reserve(new_max_elements);
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
int HNSWIndex_Multi<DataType, DistType>::deleteVector(const labelType label) {
    int ret = 0;
    // check that the label actually exists in the graph, and update the number of elements.
    auto ids = label_lookup_.find(label);
    if (ids == label_lookup_.end()) {
        return ret;
    }
    for (idType id : ids->second) {
        this->removeVector(id);
        ret++;
    }
    label_lookup_.erase(ids);
    return ret;
}

template <typename DataType, typename DistType>
int HNSWIndex_Multi<DataType, DistType>::addVector(const void *vector_data, const labelType label,
                                                   bool overwrite_allowed) {

    this->appendVector(vector_data, label);
    return 1; // We always add the vector, no overrides in multi.
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Multi<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                      VecSimQueryParams *queryParams) const {
    auto queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        normalizeVector((DataType *)queryBlobCopy, this->dim);
    }
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator) HNSWMulti_BatchIterator<DataType, DistType>(
        queryBlobCopy, this, queryParams, this->allocator);
}

/**
 * Marks an element with the given label deleted, does NOT really change the current graph.
 * @param label
 */
template <typename DataType, typename DistType>
std::vector<idType> HNSWIndex_Multi<DataType, DistType>::markDelete(labelType label) {
    std::vector<idType> idsToDelete;
    auto search = label_lookup_.find(label);
    if (search == label_lookup_.end()) {
        return idsToDelete;
    }

    for (idType id : search->second) {
        this->markDeletedInternal(id);
        idsToDelete.push_back(id);
    }
    label_lookup_.erase(search);
    return idsToDelete;
}
