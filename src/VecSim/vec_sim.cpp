#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_wrapper.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/arr_cpp.h"
#include <cassert>
#include "memory.h"

extern "C" VecSimIndex *VecSimIndex_New(const VecSimParams *params) {
    VecSimIndex *index = NULL;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    try {
        switch (params->algo) {
        case VecSimAlgo_HNSWLIB:
            index = new (allocator) HNSWIndex(&params->hnswParams, allocator);
            break;
        case VecSimAlgo_BF:
            index = new (allocator) BruteForceIndex(&params->bfParams, allocator);
            break;
        default:
            break;
        }
    } catch (...) {
        // Index will delete itself. For now, do nothing.
    }
    return index;
}

extern "C" int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id) {
    int64_t before = index->getAllocator()->getAllocationSize();
    index->addVector(blob, id);
    int64_t after = index->getAllocator()->getAllocationSize();
    return after - before;
}

extern "C" int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id) {
    int64_t before = index->getAllocator()->getAllocationSize();
    index->deleteVector(id);
    int64_t after = index->getAllocator()->getAllocationSize();
    return after - before;
}

extern "C" double VecSimIndex_GetDistanceFrom(VecSimIndex *index, size_t id, const void *blob) {
    return index->getDistanceFrom(id, blob);
}

extern "C" size_t VecSimIndex_IndexSize(VecSimIndex *index) { return index->indexSize(); }

extern "C" VecSimResolveCode VecSimIndex_ResolveParams(VecSimIndex *index, VecSimRawParam *rparams,
                                                       int paramNum, VecSimQueryParams *qparams) {
    return index->resolveParams(rparams, paramNum, qparams);
}

extern "C" VecSimQueryResult_List VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                        size_t k, VecSimQueryParams *queryParams,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    VecSimQueryResult_List results;
    results = index->topKQuery(queryBlob, k, queryParams);

    if (order == BY_ID) {
        sort_results_by_id(results);
    }
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) {
    std::shared_ptr<VecSimAllocator> allocator =
        index->getAllocator(); // Save allocator so it will not deallocate itself
    delete index;
}

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->info(); }

extern "C" VecSimInfoIterator *VecSimIndex_InfoIterator(VecSimIndex *index) {
    return index->infoIterator();
}

extern "C" VecSimBatchIterator *VecSimBatchIterator_New(VecSimIndex *index, const void *queryBlob) {
    return index->newBatchIterator(queryBlob);
}

extern "C" void VecSim_SetMemoryFunctions(VecSimMemoryFunctions memoryfunctions) {
    VecSimAllocator::setMemoryFunctions(memoryfunctions);
}
