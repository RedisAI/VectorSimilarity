/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/vec_sim_info.h"

struct HNSWInfo : public VecSimIndexInfo {
public:
    size_t M;              // Number of allowed edges per node in graph.
    size_t efConstruction; // EF parameter for HNSW graph accuracy/latency for indexing.
    size_t efRuntime;      // EF parameter for HNSW graph accuracy/latency for search.
    double epsilon;        // Epsilon parameter for HNSW graph accuracy/latency for range search.
    size_t max_level;      // Number of graph levels.
    size_t entrypoint;     // Entrypoint vector label.
    size_t visitedNodesPoolSize; // The max number of parallel graph scans so far.

    virtual VecSimInfoIterator *getIterator() override
};
