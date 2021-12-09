#pragma once

#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnswlib.h"
#include <memory>

class HNSWIndex : public VecSimIndex {
public:
    HNSWIndex(const VecSimParams *params, std::shared_ptr<VecSimAllocator> allocator);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual size_t indexSize() const override;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) override;

    void setEf(size_t ef);
    hnswlib::tableint getEntryPointId() const;

    hnswlib::HierarchicalNSW<float> hnsw;
    std::unique_ptr<SpaceInterface<float>> space;
};
