/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "vec_sim_debug.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/algorithms/hnsw/hnsw_tiered.h"
#include "VecSim/types/bfloat16.h"

extern "C" int VecSimDebug_GetElementNeighborsInHNSWGraph(VecSimIndex *index, size_t label,
                                                          int ***neighborsData) {

    // Set as if we return an error, and upon success we will set the pointers appropriately.
    *neighborsData = nullptr;
    VecSimIndexBasicInfo info = index->basicInfo();
    if (info.algo != VecSimAlgo_HNSWLIB) {
        return VecSimDebugCommandCode_BadIndex;
    }
    if (!info.isTiered) {
        if (info.type == VecSimType_FLOAT32) {
            return dynamic_cast<HNSWIndex<float, float> *>(index)->getHNSWElementNeighbors(
                label, neighborsData);
        } else if (info.type == VecSimType_FLOAT64) {
            return dynamic_cast<HNSWIndex<double, double> *>(index)->getHNSWElementNeighbors(
                label, neighborsData);
        } else if (info.type == VecSimType_BFLOAT16) {
            return dynamic_cast<HNSWIndex<vecsim_types::bfloat16, float> *>(index)
                ->getHNSWElementNeighbors(label, neighborsData);
        } else {
            assert(false && "Invalid data type");
        }
    } else {
        if (info.type == VecSimType_FLOAT32) {
            return dynamic_cast<TieredHNSWIndex<float, float> *>(index)->getHNSWElementNeighbors(
                label, neighborsData);
        } else if (info.type == VecSimType_FLOAT64) {
            return dynamic_cast<TieredHNSWIndex<double, double> *>(index)->getHNSWElementNeighbors(
                label, neighborsData);
        } else if (info.type == VecSimType_BFLOAT16) {
            return dynamic_cast<TieredHNSWIndex<vecsim_types::bfloat16, float> *>(index)
                ->getHNSWElementNeighbors(label, neighborsData);
        } else {
            assert(false && "Invalid data type");
        }
    }
    return VecSimDebugCommandCode_BadIndex;
}

extern "C" void VecSimDebug_ReleaseElementNeighborsInHNSWGraph(int **neighborsData) {
    if (neighborsData == nullptr) {
        return;
    }
    size_t i = 0;
    while (neighborsData[i] != nullptr) {
        delete[] neighborsData[i];
        i++;
    }
    delete[] neighborsData;
}
