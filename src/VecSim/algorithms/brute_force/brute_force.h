#pragma once

#include "vector_block.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include <memory>
#include <queue>
#include <cassert>
#include <limits>

class BruteForceIndex : public VecSimIndexAbstract {
protected:
    vecsim_stl::vector<labelType> idToLabelMapping;
    vecsim_stl::vector<VectorBlock *> vectorBlocks;
    idType count;
    bool multi;

public:
    BruteForceIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator);
    static BruteForceIndex *BruteForceIndex_New(const VecSimParams *params,
                                                std::shared_ptr<VecSimAllocator> allocator);
    static size_t estimateInitialSize(const BFParams *params, bool multi);
    static size_t estimateElementMemory(const BFParams *params);
    virtual size_t indexSize() const override;
    vecsim_stl::vector<float> computeBlockScores(VectorBlock *block, const void *queryBlob,
                                                 void *timeoutCtx,
                                                 VecSimQueryResult_Code *rc) const;
    virtual VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                             VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *queryBlob, float radius,
                                      VecSimQueryParams *queryParams) override;
    virtual VecSimIndexInfo info() const override;
    virtual VecSimInfoIterator *infoIterator() override;
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    virtual inline labelType getVectorId(labelType label) const = 0;
    inline labelType getVectorLabel(idType id) const { return idToLabelMapping.at(id); }
    virtual inline bool isMultiValue() const { return multi; }

    inline vecsim_stl::vector<VectorBlock *> getVectorBlocks() const { return vectorBlocks; }
    inline dist_func_t<float> distFunc() const { return dist_func; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->last_mode = mode; }
    virtual ~BruteForceIndex();

protected:
    void updateVector(idType id, const void *vector_data);
    virtual int insertVector(const void *vector_data, idType id);
    virtual int removeVector(idType id);
    inline VectorBlock *getVectorVectorBlock(idType id) { return vectorBlocks.at(id / blockSize); }
    inline size_t getVectorRelativeIndex(idType id) { return id % blockSize; }
    inline void setVectorLabel(idType id, labelType new_label) {
        idToLabelMapping.at(id) = new_label;
    }
    virtual inline void replaceIdOfLabel(labelType label, idType new_id, idType old_id) = 0;
    virtual inline void addIdToLabel(labelType label, idType id) = 0;
};
