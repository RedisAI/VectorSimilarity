#pragma once

#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnswlib.h"
#include <memory>

class HNSWIndex : public VecSimIndex {
public:
    HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator);
    static HNSWIndex *HNSWIndex_New(const HNSWParams *params, bool multi,
                                    std::shared_ptr<VecSimAllocator> allocator);
    static size_t estimateInitialSize(const HNSWParams *params, bool multi);
    static size_t estimateElementMemory(const HNSWParams *params);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual double getDistanceFrom(size_t label, const void *vector_data) override;
    virtual size_t indexSize() const override;
    virtual size_t indexLabelCount() const override;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, float radius,
                                      VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;

    void setEf(size_t ef);
    inline std::shared_ptr<hnswlib::HierarchicalNSW<float>> getHNSWIndex() { return hnsw; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->last_mode = mode; }

private:
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
};
