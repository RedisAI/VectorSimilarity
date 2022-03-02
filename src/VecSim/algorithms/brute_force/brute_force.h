#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/vecsim_stl.h"
#include <memory>
#include <queue>

class BruteForceIndex : public VecSimIndex {
protected:
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;

public:
    BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual double getDistanceFrom(size_t label, const void *vector_data) override;
    virtual size_t indexSize() const override;
    virtual VecSimResolveCode resolveParams(VecSimRawParam *rparams, int paramNum,
                                            VecSimQueryParams *qparams) override;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() override;
    virtual VecSimInfoIterator *infoIterator() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k) override;

    inline vecsim_stl::vector<VectorBlock *> getVectorBlocks() const { return vectorBlocks; }
    inline DISTFUNC<float> distFunc() const { return dist_func; }
    virtual ~BruteForceIndex();

private:
    void updateVector(idType id, const void *vector_data);
    vecsim_stl::unordered_map<labelType, idType> labelToIdLookup;
    vecsim_stl::vector<VectorBlockMember *> idToVectorBlockMemberMapping;
    vecsim_stl::set<idType> deletedIds;
    vecsim_stl::vector<VectorBlock *> vectorBlocks;
    size_t vectorBlockSize;
    idType count;
    std::unique_ptr<SpaceInterface<float>> space;
    DISTFUNC<float> dist_func;
    VecSearchMode last_mode;
#ifdef BUILD_TESTS
    // Allow the following tests to access the index size private member.
    friend class BruteForceTest_preferAdHocOptimization_Test;
    friend class BruteForceTest_test_dynamic_bf_info_iterator_Test;
#endif
};
