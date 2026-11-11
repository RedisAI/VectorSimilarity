/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/vec_sim.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace filtered_bench {

using Clock = std::chrono::steady_clock;

struct Eligibility {
    std::span<const uint64_t> ids;
    std::span<const uint8_t> membership;
};

struct Parameters {
    size_t ef_runtime;
    size_t batch_size;
    VecSimMetric metric;
    Clock::time_point deadline;
};

struct Neighbor {
    size_t id;
    double distance;
};

struct SearchResult {
    std::vector<Neighbor> neighbors;
    std::string status = "ok";
    std::optional<size_t> distance_evaluations;
    std::optional<size_t> candidates_returned;
    std::optional<size_t> batches;
};

// Adapters receive the same loaded graph and eligibility set. Query preprocessing,
// traversal, filtering, top-k selection, and adapter cleanup belong inside this call.
using SearchFunction = SearchResult (*)(VecSimIndex *, std::span<const float>, const Eligibility &,
                                        size_t, const Parameters &);

struct CandidateAdapter {
    std::string name;
    SearchFunction search;
};

// Register an implemented candidate here; a label alone is not an implementation.
std::vector<CandidateAdapter> candidateAdapters();
int deadlineExpired(void *context);

} // namespace filtered_bench
