/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/vec_sim_info.h"

struct BruteForceInfo : public VecSimInfo {
public:
    BruteForceInfo(VecSimInfo *info);
    size_t blockSize; // Brute force algorithm vector block (mini matrix) size
    virtual VecSimInfoIterator *getIterator() override;
};
