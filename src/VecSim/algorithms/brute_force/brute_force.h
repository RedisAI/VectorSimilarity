#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include <memory>
#include <queue>
#include <cassert>
#include <limits>

class BruteForceIndex : public VecSimIndex {
protected:
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;

public:
    BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    static size_t estimateInitialSize(const BFParams *params);
    static size_t estimateElementMemory(const BFParams *params);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual double getDistanceFrom(size_t label, const void *vector_data) override;
    virtual size_t indexSize() const override;
    vecsim_stl::vector<float> computeBlockScores(VectorBlock *block, const void *queryBlob,
                                                 void *timeoutCtx,
                                                 VecSimQueryResult_Code *rc) const;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, float radius,
                                      VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    inline labelType getVectorLabel(idType id) const {
        return idToLabelMapping.at(id);
    } // throws out_of_range
    inline labelType getVectorId(labelType label) const {
        return labelToIdLookup.at(label);
    } // throws out_of_range

    inline vecsim_stl::vector<VectorBlock *> getVectorBlocks() const { return vectorBlocks; }
    inline Spaces::dist_func_t<float> distFunc() const { return dist_func; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->last_mode = mode; }
    virtual ~BruteForceIndex();

private:
    void updateVector(idType id, const void *vector_data);
    inline VectorBlock *getVectorVectorBlock(idType id) {
        return vectorBlocks.at(id / vectorBlockSize);
    }
    inline size_t getVectorRelativeIndex(idType id) { return id % vectorBlockSize; }
    inline void setVectorLabel(idType id, labelType new_label) {
        idToLabelMapping.at(id) = new_label;
    } // throws out_of_range
    inline void setLabelToId(labelType label, idType new_id) {
        labelToIdLookup.at(label) = new_id;
    } // throws out_of_range

    vecsim_stl::unordered_map<labelType, idType> labelToIdLookup;
    vecsim_stl::vector<labelType> idToLabelMapping;
    vecsim_stl::vector<VectorBlock *> vectorBlocks;
    size_t vectorBlockSize;
    idType count;
    Spaces::dist_func_t<float> dist_func;
    VecSearchMode last_mode;
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
