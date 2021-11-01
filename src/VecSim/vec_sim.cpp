#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnswlib_c.h"
#include "VecSim/utils/arr_cpp.h"
#include <cassert>
#include "VecSim/utils/vec_utils.h"
#include "memory.h"

int cmpVecSimQueryResult(const VecSimQueryResult *res1, const VecSimQueryResult *res2) {
    return res1->id > res2->id ? 1 : res1->id < res2->id ? -1 : 0;
}
extern "C" VecSimIndex *VecSimIndex_New(const VecSimParams *params) {
    std::shared_ptr<VecSimAllocator> allocator = std::make_shared<VecSimAllocator>();
    if (params->algo == VecSimAlgo_HNSWLIB) {
        return new (allocator) HNSWIndex(params, allocator);
    }
    return new (allocator) BruteForceIndex(params, allocator);
}

extern "C" int VecSimIndex_AddVector(VecSimIndex *index, const void *blob, size_t id) {
    if (index->getMetric() == VecSimMetric_Cosine) {
        // TODO: need more generic
        float normalized_blob[index->getVectorDim()];
        memcpy(normalized_blob, blob, index->getVectorDim() * sizeof(float));
        float_vector_normalize(normalized_blob, index->getVectorDim());
        return index->addVector(normalized_blob, id);
    }
    return index->addVector(blob, id);
}

extern "C" int VecSimIndex_DeleteVector(VecSimIndex *index, size_t id) {
    return index->deleteVector(id);
}

extern "C" size_t VecSimIndex_IndexSize(VecSimIndex *index) { return index->indexSize(); }

extern "C" VecSimQueryResult_List VecSimIndex_TopKQuery(VecSimIndex *index, const void *queryBlob,
                                                        size_t k, VecSimQueryParams *queryParams,
                                                        VecSimQueryResult_Order order) {
    assert((order == BY_ID || order == BY_SCORE) &&
           "Possible order values are only 'BY_ID' or 'BY_SCORE'");
    VecSimQueryResult_List results;
    if (index->getMetric() == VecSimMetric_Cosine) {
        // TODO: need more generic
        float normalized_blob[index->getVectorDim()];
        memcpy(normalized_blob, queryBlob, index->getVectorDim() * sizeof(float));
        float_vector_normalize(normalized_blob, index->getVectorDim());
        results = index->topKQuery(normalized_blob, k, queryParams);
    } else {
        results = index->topKQuery(queryBlob, k, queryParams);
    }
    if (order == BY_ID) {
        // sort results by id and then return.
        qsort(results, VecSimQueryResult_Len(results), sizeof(VecSimQueryResult),
              (__compar_fn_t)cmpVecSimQueryResult);
    }
    return results;
}

extern "C" void VecSimIndex_Free(VecSimIndex *index) { delete index; }

extern "C" VecSimIndexInfo VecSimIndex_Info(VecSimIndex *index) { return index->info(); }
