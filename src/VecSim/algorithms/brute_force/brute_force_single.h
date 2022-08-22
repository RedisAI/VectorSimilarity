#pragma once

#include "brute_force.h"
#include "bfs_batch_iterator.h"

class BruteForceIndex_Single : public BruteForceIndex {
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

    inline vecsim_stl::abstract_priority_queue<float, labelType> *getNewPriorityQueue() override {
        return new (this->allocator)
            vecsim_stl::max_priority_queue<float, labelType>(this->allocator);
    }

    inline BF_BatchIterator *newBatchIterator_Instance(void *queryBlob,
                                                       VecSimQueryParams *queryParams) override {
        return new (allocator) BFS_BatchIterator(queryBlob, this, queryParams, allocator);
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
