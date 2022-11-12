#pragma once

#include "ngt.h"
// #include "ngt_single_batch_iterator.h"

template <typename DataType, typename DistType>
class NGTIndex_Single : public NGTIndex<DataType, DistType> {
private:
    vecsim_stl::unordered_map<labelType, idType> label_lookup_;

#ifdef BUILD_TESTS
//     friend class HNSWIndexSerializer;
#include "VecSim/algorithms/ngt/ngt_single_tests_friends.h"
#endif

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override { label_lookup_[label] = id; }
    inline void resizeLabelLookup(size_t new_max_elements) override;

public:
    NGTIndex_Single(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
                    size_t random_seed = 100, size_t initial_pool_size = 1)
        : NGTIndex<DataType, DistType>(params, allocator, random_seed, initial_pool_size),
          label_lookup_(this->max_elements_, allocator) {}

    ~NGTIndex_Single() {}

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
    int addVector(const void *vector_data, labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
};

/**
 * getters and setters of index data
 */

template <typename DataType, typename DistType>
size_t NGTIndex_Single<DataType, DistType>::indexLabelCount() const {
    return label_lookup_.size();
}

template <typename DataType, typename DistType>
double NGTIndex_Single<DataType, DistType>::getDistanceFrom(labelType label,
                                                            const void *vector_data) const {
    auto id = label_lookup_.find(label);
    if (id == label_lookup_.end()) {
        return INVALID_SCORE;
    }
    return this->dist_func(vector_data, this->getDataByInternalId(id->second), this->dim);
}

/**
 * helper functions
 */

template <typename DataType, typename DistType>
void NGTIndex_Single<DataType, DistType>::replaceIdOfLabel(labelType label, idType new_id,
                                                           idType old_id) {
    label_lookup_[label] = new_id;
}

template <typename DataType, typename DistType>
void NGTIndex_Single<DataType, DistType>::resizeLabelLookup(size_t new_max_elements) {
    label_lookup_.reserve(new_max_elements);
}

/**
 * Index API functions
 */

template <typename DataType, typename DistType>
int NGTIndex_Single<DataType, DistType>::deleteVector(const labelType label) {
    // check that the label actually exists in the graph, and update the number of elements.
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return false;
    }
    idType element_internal_id = label_lookup_[label];
    label_lookup_.erase(label);
    return this->removeVector(element_internal_id);
}

template <typename DataType, typename DistType>
int NGTIndex_Single<DataType, DistType>::addVector(const void *vector_data, const labelType label) {

    // Checking if an element with the given label already exists. if so, remove it.
    if (label_lookup_.find(label) != label_lookup_.end()) {
        deleteVector(label);
    }

    return this->appendVector(vector_data, label);
}

template <typename DataType, typename DistType>
VecSimBatchIterator *
NGTIndex_Single<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                      VecSimQueryParams *queryParams) const {
    auto queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        normalizeVector((DataType *)queryBlobCopy, this->dim);
    }
    // Ownership of queryBlobCopy moves to NGT_BatchIterator that will free it at the end.
    return nullptr;
}
