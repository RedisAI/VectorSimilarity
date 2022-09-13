#pragma once

#include "hnsw.h"
#include "hnsw_single_batch_iterator.h"

template <typename DataType, typename DistType>
class HNSWIndex_Single : public HNSWIndex<DataType, DistType> {
private:
    vecsim_stl::unordered_map<labelType, idType> label_lookup_;

#ifdef BUILD_TESTS
    friend class HNSWIndexSerializer;
    // Allow the following test to access the index size private member.
    friend class AllocatorTest_testIncomingEdgesSet_Test;
    friend class AllocatorTest_test_hnsw_reclaim_memory_Test;
    friend class HNSWTest_testSizeEstimation_Test;
    friend class HNSWTest_test_dynamic_hnsw_info_iterator_Test;
    friend class HNSWTest_preferAdHocOptimization_Test;
#endif

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;
    inline void setVectorId(labelType label, idType id) override { label_lookup_[label] = id; }
    inline void resizeLabelLookup(size_t new_max_elements) override;

public:
    HNSWIndex_Single(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
                     size_t random_seed = 100, size_t initial_pool_size = 1)
        : HNSWIndex<DataType, DistType>(params, allocator, random_seed, initial_pool_size),
          label_lookup_(this->max_elements_, allocator) {}

    ~HNSWIndex_Single() {}

    inline candidatesLabelsMaxHeap<DistType> *getNewMaxPriorityQueue() const override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
    }

    inline size_t indexLabelCount() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) override;

    int deleteVector(labelType label) override;
    int addVector(const void *vector_data, labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
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
    if (id == label_lookup_.end()) {
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
    // check that the label actually exists in the graph, and update the number of elements.
    if (label_lookup_.find(label) == label_lookup_.end()) {
        return false;
    }
    idType element_internal_id = label_lookup_[label];
    label_lookup_.erase(label);
    return this->removeVector(element_internal_id);
}

template <typename DataType, typename DistType>
int HNSWIndex_Single<DataType, DistType>::addVector(const void *vector_data,
                                                    const labelType label) {

    // Checking if an element with the given label already exists. if so, remove it.
    if (label_lookup_.find(label) != label_lookup_.end()) {
        deleteVector(label);
    }

    return this->appendVector(vector_data, label);
}


template <typename DataType, typename DistType>
VecSimBatchIterator *
HNSWIndex_Single<DataType, DistType>::newBatchIterator(const void *queryBlob,
                                                       VecSimQueryParams *queryParams) {
    // As this is the only supported type, we always allocate 4 bytes for every element in the
    // vector.
    assert(this->vecType == VecSimType_FLOAT32);
    auto queryBlobCopy = this->allocator->allocate(sizeof(DataType) * this->dim);
    memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
    if (this->metric == VecSimMetric_Cosine) {
        normalizeVector((DataType *)queryBlobCopy, this->dim);
    }
    // Ownership of queryBlobCopy moves to HNSW_BatchIterator that will free it at the end.
    return new (this->allocator) HNSWSingle_BatchIterator<DataType, DistType>(
        queryBlobCopy, this, queryParams, this->allocator);
}
