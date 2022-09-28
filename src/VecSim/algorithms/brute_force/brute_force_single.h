#pragma once

#include "brute_force.h"
#include "bfs_batch_iterator.h"
#include "VecSim/utils/vec_utils.h"

template <typename DataType, typename DistType>
class BruteForceIndex_Single : public BruteForceIndex<DataType, DistType> {

protected:
    vecsim_stl::unordered_map<labelType, idType> labelToIdLookup;

public:
    BruteForceIndex_Single(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    ~BruteForceIndex_Single();

    int addVector(const void *vector_data, labelType label) override;
    int deleteVector(labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;

    inline size_t indexLabelCount() const override { return this->count; }

protected:
    // inline definitions

    inline void updateVector(idType id, const void *vector_data) {

        // Get the vector block
        VectorBlock *vectorBlock = this->getVectorVectorBlock(id);
        size_t index = BruteForceIndex<DataType, DistType>::getVectorRelativeIndex(id);

        // Update vector data in the block.
        vectorBlock->updateVector(index, vector_data);
    }

    inline void setVectorId(labelType label, idType id) override {
        labelToIdLookup.emplace(label, id);
    }

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override {
        labelToIdLookup.at(label) = new_id;
    }

    inline vecsim_stl::abstract_priority_queue<DistType, labelType> *
    getNewMaxPriorityQueue() override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<DistType, labelType>(this->allocator);
    }

    inline BF_BatchIterator<DataType, DistType> *
    newBatchIterator_Instance(void *queryBlob, VecSimQueryParams *queryParams) const override {
        return new (this->allocator)
            BFS_BatchIterator<DataType, DistType>(queryBlob, this, queryParams, this->allocator);
    }

#ifdef BUILD_TESTS
    template <VecSimType type, typename DataType, typename DistType = DataType>
    struct IndexType;
    template <typename index_type_t>
    class BruteForceTest ;
    FRIEND_TEST(BruteForceTest, brute_force_reindexing_same_vector);
    // Allow the following tests to access the index private members.
    friend class BruteForceTest_preferAdHocOptimization_Test;
    friend class BruteForceTest_test_dynamic_bf_info_iterator_Test;
    friend class BruteForceTest_resize_and_align_index_Test;
    friend class BruteForceTest_brute_force_vector_update_test_Test;
   // friend class BruteForceTest_brute_force_reindexing_same_vector_Test;
    friend class BruteForceTest_test_delete_swap_block_Test;
    friend class BruteForceTest_brute_force_zero_minimal_capacity_Test;
    friend class BruteForceTest_resize_and_align_index_largeInitialCapacity_Test;
    friend class BruteForceTest_brute_force_empty_index_Test;
    friend class BM_VecSimBasics_DeleteVectorBF_Benchmark;
#endif
};

/******************************* Implementation **********************************/

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::BruteForceIndex_Single(
    const BFParams *params, std::shared_ptr<VecSimAllocator> allocator)
    : BruteForceIndex<DataType, DistType>(params, allocator), labelToIdLookup(allocator) {}

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::~BruteForceIndex_Single() {}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::addVector(const void *vector_data,
                                                          labelType label) {

    DataType normalized_blob[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        memcpy(normalized_blob, vector_data, this->dim * sizeof(DataType));
        normalizeVector(normalized_blob, this->dim);
        vector_data = normalized_blob;
    }

    auto optionalID = this->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != this->labelToIdLookup.end()) {
        idType id = optionalID->second;
        updateVector(id, vector_data);
        return true;
    }

    return this->appendVector(vector_data, label);
}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::deleteVector(labelType label) {

    // Find the id to delete.
    auto deleted_label_id_pair = this->labelToIdLookup.find(label);
    if (deleted_label_id_pair == this->labelToIdLookup.end()) {
        // Nothing to delete.
        return true;
    }

    // Get deleted vector id.
    idType id_to_delete = deleted_label_id_pair->second;

    // Remove the pair of the deleted vector.
    labelToIdLookup.erase(label);

    return this->removeVector(id_to_delete);
}

template <typename DataType, typename DistType>
double BruteForceIndex_Single<DataType, DistType>::getDistanceFrom(labelType label,
                                                                   const void *vector_data) const {

    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = optionalId->second;

    return this->dist_func(this->getDataByInternalId(id), vector_data, this->dim);
}
