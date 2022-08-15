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
    virtual double getDistanceFrom(size_t label, const void *vector_data) const override;

    virtual inline size_t indexLabelCount() const override { return this->labelToIdsLookup.size(); }

private:
    inline void setVectorId(labelType label, idType id) override;
    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override;

#ifdef BUILD_TESTS
    // Allow the following tests to access the index private members.
    friend class BruteForceMultiTest_resizeNAlignIndex_Test;
    friend class BruteForceMultiTest_brute_force_empty_index_Test;
    friend class BruteForceMultiTest_test_delete_swap_block_Test;
#endif
};
