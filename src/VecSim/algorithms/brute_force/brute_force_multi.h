#pragma once

#include "brute_force.h"

class BruteForceIndex_Multi : public BruteForceIndex {
private:
    vecsim_stl::unordered_map<labelType, vecsim_stl::vector<idType>> labelToIdsLookup;

public:
    BruteForceIndex_Multi(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    ~BruteForceIndex_Multi();

    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual double getDistanceFrom(size_t label, const void *vector_data) override;

    virtual inline size_t indexLabelCount() const override { return this->labelToIdsLookup.size(); }

private:
    inline void addIdToLabel(labelType label, idType id) override;
    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;

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
