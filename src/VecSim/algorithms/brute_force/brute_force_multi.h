#pragma once

#include "brute_force.h"
#include "VecSim/utils/updatable_heap.h"

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
    // inline definitions

    inline void setVectorId(labelType label, idType id) override {
        auto labelKey = labelToIdsLookup.find(label);
        if (labelKey != labelToIdsLookup.end()) {
            labelKey->second.push_back(id);
        } else {
            // Initial capacity is 1. We can consider increasing this value or having it as a
            // parameter.
            labelToIdsLookup.emplace(label, vecsim_stl::vector<idType>{1, id, this->allocator});
        }
    }

    inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) override {
        assert(labelToIdsLookup.find(label) != labelToIdsLookup.end());
        auto &labelKey = labelToIdsLookup.at(label);
        for (size_t i = 0; i < labelKey.size(); i++) {
            if (labelKey[i] == old_id) {
                labelKey[i] = new_id;
                return;
            }
        }
        assert(false && "should have find the old id");
    }

    inline vecsim_stl::priority_queue_abstract<labelType, float> *getNewPriorityQueue() override {
        return new (this->allocator)
            vecsim_stl::updatable_max_heap<labelType, float>(this->allocator);
    }

#ifdef BUILD_TESTS
    // Allow the following tests to access the index private members.
    friend class BruteForceMultiTest_resizeNAlignIndex_Test;
    friend class BruteForceMultiTest_empty_index_Test;
    friend class BruteForceMultiTest_test_delete_swap_block_Test;
    friend class BruteForceMultiTest_remove_vector_after_replacing_block_Test;
    friend class BruteForceMultiTest_search_more_then_there_is_Test;
    friend class BruteForceMultiTest_indexing_same_vector_Test;
#endif
};
