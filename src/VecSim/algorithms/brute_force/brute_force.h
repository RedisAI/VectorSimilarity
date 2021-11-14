#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/space_interface.h"
#include "VecSim/utils/vecsim_stl.h"
#include <memory>

class BruteForceIndex : public VecSimIndex {
public:
    BruteForceIndex(const VecSimParams *params, std::shared_ptr<VecSimAllocator> allocator);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual size_t indexSize() override;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) override;

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
};
