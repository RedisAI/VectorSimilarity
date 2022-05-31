#pragma once

#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnswlib.h"
#include <memory>

class HNSWIndex : public VecSimIndex {
protected:
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
    size_t blockSize;

public:
    HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator);
    static size_t estimateInitialSize(const HNSWParams *params);
    static size_t estimateElementMemory(const HNSWParams *params);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual double getDistanceFrom(size_t label, const void *vector_data) override;
    virtual size_t indexSize() const override;
    virtual VecSimResolveCode resolveParams(VecSimRawParam *rparams, int paramNum,
                                            VecSimQueryParams *qparams) override;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;

    void setEf(size_t ef);
    inline std::shared_ptr<hnswlib::HierarchicalNSW<float>> getHNSWIndex() { return hnsw; }
    inline std::shared_ptr<SpaceInterface<float>> getSpace() { return space; }

private:
    std::shared_ptr<SpaceInterface<float>> space;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw;
    VecSearchMode last_mode;
};
