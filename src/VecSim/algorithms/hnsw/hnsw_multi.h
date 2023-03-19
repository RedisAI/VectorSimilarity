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
    // Index global state - this should be guarded by the index_data_guard_ lock in
    // multithreaded scenario.
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

    template <bool Safe>
    inline double getDistanceFromInternal(labelType label, const void *vector_data) const;

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

    void getDataByLabel(labelType label,
                        std::vector<std::vector<DataType>> &vectors_output) const override {

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
    int addVector(const void *vector_data, labelType label,
                  idType new_vec_id = INVALID_ID) override;
    inline std::vector<idType> markDelete(labelType label) override;
    inline bool safeCheckIfLabelExistsInIndex(labelType label,
                                              bool also_done_processing) const override;
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
size_t HNSWIndex_Multi<DataType, DistType>::indexLabelCount() const {
    return label_lookup_.size();
}

/**
 * helper functions
 */

// Depending on the value of the Safe template parameter, this function will either return a copy
// of the argument or a reference to it.
template <bool Safe, typename Arg>
constexpr decltype(auto) getCopyOrReference(Arg &&arg) {
    if constexpr (Safe) {
        return std::decay_t<Arg>(arg);
    } else {
        return (arg);
    }
}

template <typename DataType, typename DistType>
template <bool Safe>
double HNSWIndex_Multi<DataType, DistType>::getDistanceFromInternal(labelType label,
                                                                    const void *vector_data) const {
    DistType dist = INVALID_SCORE;

    // Check if the label exists in the index, return invalid score if not.
    if (Safe)
        this->index_data_guard_.lock_shared();
    auto it = this->label_lookup_.find(label);
    if (it == this->label_lookup_.end()) {
        if (Safe)
            this->index_data_guard_.unlock_shared();
        return dist;
    }

    // Get the vector of ids associated with the label.
    // Get a copy if `Safe` is true, otherwise get a reference.
    decltype(auto) IDs = getCopyOrReference<Safe>(it->second);
    if (Safe)
        this->index_data_guard_.unlock_shared();

    // Iterate over the ids and find the minimum distance.
    for (auto id : IDs) {
        DistType d = this->dist_func(this->getDataByInternalId(id), vector_data, this->dim);
        dist = std::fmin(dist, d);
    }

    return dist;
}

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
                                                   idType new_vec_id) {

    this->appendVector(vector_data, label, new_vec_id);
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

template <typename DataType, typename DistType>
inline bool HNSWIndex_Multi<DataType, DistType>::safeCheckIfLabelExistsInIndex(
    labelType label, bool also_done_processing) const {
    std::unique_lock<std::shared_mutex> index_data_lock(this->index_data_guard_);
    auto search_res = label_lookup_.find(label);
    bool exists = search_res != label_lookup_.end();
    // If we want to make sure that the vector(s) stored under the label were already indexed,
    // we go on and check that every associated vector is no longer in process.
    if (exists && also_done_processing) {
        for (auto id : search_res->second) {
            exists = !this->isInProcess(id);
            // If we find at least one internal id that is still in process, consider it as not
            // ready.
            if (!exists)
                return false;
        }
    }
    return exists;
}
