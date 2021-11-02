#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/space_interface.h"
#include <unordered_map>
#include <set>
#include <vector>
#include <memory>
#include <queue>


class BruteForceIndex : public VecSimIndex {
public:
    BruteForceIndex(const VecSimParams *params);
    virtual int addVector(const void *vector_data, size_t label) override;
    virtual int deleteVector(size_t id) override;
    virtual size_t indexSize() const;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob) override;
    inline std::vector<VectorBlock *> getVectorBlocks() const {
        return vectorBlocks;
    }
    inline DISTFUNC<float> distFunc() const {
        return dist_func;
    }
    virtual ~BruteForceIndex();

private:
    void updateVector(idType id, const void *vector_data);
    std::unordered_map<labelType, idType> labelToIdLookup;
    std::vector<VectorBlockMember *> idToVectorBlockMemberMapping;
    std::set<idType> deletedIds;
    std::vector<VectorBlock *> vectorBlocks;
    size_t vectorBlockSize;
    idType count;
    std::unique_ptr<SpaceInterface<float>> space;
    DISTFUNC<float> dist_func;
};
