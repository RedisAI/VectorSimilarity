/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/info/vec_sim_info.h"

struct TieredIndexInfo : public VecSimIndexInfo {
public:
    VecSimIndexInfo *backendIndexInfo;
    VecSimIndexInfo *frontendIndexInfo;

    size_t management_layer_memory;

    virtual VecSimInfoIterator *getIterator();
};
