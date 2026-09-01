/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "vec_sim_debug.h"
#include "VecSim/vec_sim_index.h"

extern "C" int VecSimDebug_GetElementNeighborsInHNSWGraph(VecSimIndex *index, size_t label,
                                                          int ***neighborsData) {
    return index->getHNSWElementNeighbors(label, neighborsData);
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
