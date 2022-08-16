#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include <memory>
#include <queue>
#include <cassert>
#include <limits>

class BruteForceIndex : public VecSimIndexAbstract {
protected:
    vecsim_stl::vector<labelType> idToLabelMapping;
    vecsim_stl::vector<VectorBlock *> vectorBlocks;
    idType count;

public:
    BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    static BruteForceIndex *BruteForceIndex_New(const BFParams *params,
                                                std::shared_ptr<VecSimAllocator> allocator);
    static size_t estimateInitialSize(const BFParams *params);
    static size_t estimateElementMemory(const BFParams *params);
    virtual size_t indexSize() const override;
    vecsim_stl::vector<float> computeBlockScores(VectorBlock *block, const void *queryBlob,
                                                 void *timeoutCtx,
                                                 VecSimQueryResult_Code *rc) const;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, float radius,
                                      VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() const override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    inline labelType getVectorLabel(idType id) const { return idToLabelMapping.at(id); }

    inline vecsim_stl::vector<VectorBlock *> getVectorBlocks() const { return vectorBlocks; }
    virtual ~BruteForceIndex();

protected:
    // Private internal function that implements generic single vector insertion.
    virtual int appendVector(const void *vector_data, labelType label);

    // Private internal function that implements generic single vector deletion.
    virtual int removeVector(idType id);

    inline float *getDataByInternalId(idType id) const {
        return vectorBlocks.at(id / blockSize)->getVector(id % blockSize);
    }
    inline VectorBlock *getVectorVectorBlock(idType id) const {
        return vectorBlocks.at(id / blockSize);
    }
    inline size_t getVectorRelativeIndex(idType id) const { return id % blockSize; }
    inline void setVectorLabel(idType id, labelType new_label) {
        idToLabelMapping.at(id) = new_label;
    }
    // inline priority queue getter that need to be implemented by derived class
    virtual inline vecsim_stl::priority_queue_abstract<labelType, float> *getNewPriorityQueue() = 0;

    // inline label to id setters that need to be implemented by derived class
    virtual inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) = 0;
    virtual inline void setVectorId(labelType label, idType id) = 0;

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
