#pragma once

#include "brute_force.h"
#include "bfm_batch_iterator.h"
#include "VecSim/utils/updatable_heap.h"

template <typename DataType, typename DistType>
class BruteForceIndex_Multi : public BruteForceIndex<DataType, DistType> {
private:
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>> labelToIdsLookup;

public:
    BruteForceIndex_Multi(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
        : BruteForceIndex<DataType, DistType>(params, allocator), labelToIdsLookup(allocator) {}

    ~BruteForceIndex_Multi() {}

    int addVector(const void *vector_data, labelType label) override;
    int deleteVector(labelType labelType) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;

    inline size_t indexLabelCount() const override { return this->labelToIdsLookup.size(); }

private:
    // inline definitions

    inline void setVectorId(labelType label, idType id) override;

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;

    inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() override {
        return new (this->allocator)
            vecsim_stl::updatable_max_heap<DistType, labelType>(this->allocator);
    }

    inline BF_BatchIterator *newBatchIterator_Instance(void *queryBlob,
                                                       VecSimQueryParams *queryParams) override {
        return new (this->allocator)
            BFM_BatchIterator(queryBlob, this, queryParams, this->allocator);
    }

#ifdef BUILD_TESTS
    // Allow the following tests to access the index private members.
    friend class BruteForceMultiTest_resize_and_align_index_Test;
    friend class BruteForceMultiTest_empty_index_Test;
    friend class BruteForceMultiTest_test_delete_swap_block_Test;
    friend class BruteForceMultiTest_remove_vector_after_replacing_block_Test;
    friend class BruteForceMultiTest_search_more_then_there_is_Test;
    friend class BruteForceMultiTest_indexing_same_vector_Test;
    friend class BruteForceMultiTest_test_dynamic_bf_info_iterator_Test;
#endif
};

/******************************* Implementation **********************************/

template <typename DataType, typename DistType>
int BruteForceIndex_Multi<DataType, DistType>::addVector(const void *vector_data, labelType label) {

    DataType normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, vector_data, this->dim * sizeof(DataType));
        float_vector_normalize(normalized_data, this->dim);
        vector_data = normalized_data;
    }

    return this->appendVector(vector_data, label);
}

template <typename DataType, typename DistType>
int BruteForceIndex_Multi<DataType, DistType>::deleteVector(labelType label) {

    // Find the id to delete.
    auto deleted_label_ids_pair = this->labelToIdsLookup.find(label);
    if (deleted_label_ids_pair == this->labelToIdsLookup.end()) {
        // Nothing to delete.
        return true;
    }

    int ret = true;

    // Deletes all vectors under the given label.
    for (auto id_to_delete : deleted_label_ids_pair->second) {
        ret = (this->removeVector(id_to_delete) && ret);
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
