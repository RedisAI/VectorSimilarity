#pragma once

#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnswlib.h"
#include <memory>

class HNSWIndex : public VecSimIndex {
protected:
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;

public:
    HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual size_t indexSize() const override;
    virtual VecSimResolveCode resolveParams(VecSimRawParam *rparams, int paramNum, VecSimQueryParams *qparams) override;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() override;
    virtual VecSimInfoIterator *infoIterator() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) override;

    void setEf(size_t ef);
    inline std::shared_ptr<hnswlib::HierarchicalNSW<float>> getHNSWIndex() { return hnsw; }
    inline std::shared_ptr<SpaceInterface<float>> getSpace() { return space; }

private:
    std::shared_ptr<SpaceInterface<float>> space;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
};
