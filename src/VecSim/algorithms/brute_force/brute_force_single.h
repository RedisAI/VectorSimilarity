#pragma once

#include "brute_force.h"
template <typename DataType, typename DistType>
class BruteForceIndex_Single : public BruteForceIndex<DataType, DistType> {
protected:
    vecsim_stl::unordered_map<labelType, idType> labelToIdLookup;

public:
    BruteForceIndex_Single(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    ~BruteForceIndex_Single();

    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual double getDistanceFrom(size_t label, const void *vector_data) const override;

    virtual inline size_t indexLabelCount() const override { return this->count; }

protected:
    // inline definitions

    inline void updateVector(idType id, const void *vector_data) {

        // Get the vector block
        VectorBlock *vectorBlock = getVectorVectorBlock(id);
        size_t index = getVectorRelativeIndex(id);

        // Update vector data in the block.
        vectorBlock->updateVector(index, vector_data);
    }

    inline void setVectorId(labelType label, idType id) override {
        labelToIdLookup.emplace(label, id);
    }

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override {
        labelToIdLookup.at(label) = new_id;
    }

#ifdef BUILD_TESTS
    // Allow the following tests to access the index private members.
    friend class BruteForceTest_preferAdHocOptimization_Test;
    friend class BruteForceTest_test_dynamic_bf_info_iterator_Test;
    friend class BruteForceTest_resizeNAlignIndex_Test;
    friend class BruteForceTest_brute_force_vector_update_test_Test;
    friend class BruteForceTest_brute_force_reindexing_same_vector_Test;
    friend class BruteForceTest_test_delete_swap_block_Test;
    friend class BruteForceTest_brute_force_zero_minimal_capacity_Test;
    friend class BruteForceTest_resizeNAlignIndex_largeInitialCapacity_Test;
    friend class BruteForceTest_brute_force_empty_index_Test;
    friend class BM_VecSimBasics_DeleteVectorBF_Benchmark;
#endif
};


/******************************* Implementation **********************************/

#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::BruteForceIndex_Single(const BFParams *params,
                                               std::shared_ptr<VecSimAllocator> allocator)
    : BruteForceIndex(params, allocator), labelToIdLookup(allocator) {}

template <typename DataType, typename DistType>
BruteForceIndex_Single<DataType, DistType>::~BruteForceIndex_Single() {}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::addVector(const void *vector_data, size_t label) {

    DataType normalized_data[this->dim]; // This will be use only if metric == VecSimMetric_Cosine
    if (this->metric == VecSimMetric_Cosine) {
        // TODO: need more generic
        memcpy(normalized_data, vector_data, this->dim * sizeof(DataType));
        float_vector_normalize(normalized_data, this->dim);
        vector_data = normalized_data;
    }

    auto optionalID = this->labelToIdLookup.find(label);
    // Check if label already exists, so it is an update operation.
    if (optionalID != this->labelToIdLookup.end()) {
        idType id = optionalID->second;
        updateVector(id, vector_data);
        return true;
    }

    return appendVector(vector_data, label);
}

template <typename DataType, typename DistType>
int BruteForceIndex_Single<DataType, DistType>::deleteVector(size_t label) {

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

    return removeVector(id_to_delete);
}

template <typename DataType, typename DistType>
double BruteForceIndex_Single<DataType, DistType>::getDistanceFrom(size_t label, const void *vector_data) const {

    auto optionalId = this->labelToIdLookup.find(label);
    if (optionalId == this->labelToIdLookup.end()) {
        return INVALID_SCORE;
    }
    idType id = optionalId->second;

    return this->dist_func(getDataByInternalId(id), vector_data, this->dim);
}
