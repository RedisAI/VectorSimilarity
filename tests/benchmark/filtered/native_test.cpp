/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "native_candidates.h"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace filtered_bench;

namespace {

size_t timeout_callback_calls = 0;

int forceTimeout(void *) {
    ++timeout_callback_calls;
    return 1;
}

struct RestoreTimeoutCallback {
    ~RestoreTimeoutCallback() { VecSim_SetTimeoutCallbackFunction(deadlineExpired); }
};

void check(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

SearchFunction adapter(const std::string &name) {
    for (const auto &candidate : candidateAdapters()) {
        if (candidate.name == name) {
            return candidate.search;
        }
    }
    throw std::runtime_error("Missing native adapter: " + name);
}

void checkRecovered(const SearchResult &result, const std::string &name) {
    check(result.status == "ok", name + " did not recover after timeout");
    check(result.neighbors.size() == 3, name + " returned the wrong number of neighbors");
    for (size_t i = 0; i < result.neighbors.size(); ++i) {
        const size_t expected_id = 2 * i;
        check(result.neighbors[i].id == expected_id,
              name + " returned the wrong eligible neighbor after timeout");
        check(std::isfinite(result.neighbors[i].distance) &&
                  result.neighbors[i].distance == double(expected_id * expected_id),
              name + " returned the wrong distance after timeout");
    }
}

void testTimeoutsAndRecovery() {
    constexpr size_t documents = 40;
    constexpr size_t dimensions = 2;
    VecSimParams params = {};
    params.algo = VecSimAlgo_HNSWLIB;
    auto &hnsw = params.algoParams.hnswParams;
    hnsw.type = VecSimType_FLOAT32;
    hnsw.dim = dimensions;
    hnsw.metric = VecSimMetric_L2;
    hnsw.M = 8;
    hnsw.efConstruction = documents;
    hnsw.efRuntime = documents;
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&params),
                                                                    &VecSimIndex_Free);
    check(bool(index), "Could not create the test HNSW index");
    std::vector<uint64_t> eligible_ids;
    std::vector<uint8_t> membership(documents, 0);
    for (size_t id = 0; id < documents; ++id) {
        const std::array<float, dimensions> vector{float(id), 0.0f};
        check(VecSimIndex_AddVector(index.get(), vector.data(), id) == 1,
              "Could not insert a test vector");
        if (id % 2 == 0) {
            eligible_ids.push_back(id);
            membership[id] = 1;
        }
    }
    const Eligibility eligible{eligible_ids, membership};
    const std::array<float, dimensions> query{0.0f, 0.0f};
    const auto scan = adapter("NATIVE_SCAN");
    const auto batches = adapter("NATIVE_BATCHES");
    Parameters runtime{documents, 4, VecSimMetric_L2, Clock::now() - std::chrono::seconds(1)};
    VecSim_SetTimeoutCallbackFunction(deadlineExpired);
    const auto expired_scan = scan(index.get(), query, eligible, 3, runtime);
    check(expired_scan.status == "timeout" && expired_scan.neighbors.empty(),
          "An expired scan deadline must return timeout with no results");
    check(expired_scan.distance_evaluations == 0,
          "An already expired scan must not calculate distances");

    // Keep the runner's deadline in the future so this exercises a VecSim reply
    // timeout, rather than the adapter's own pre-batch deadline check.
    runtime.deadline = Clock::now() + std::chrono::hours(1);
    SearchResult timed_out_batch;
    {
        RestoreTimeoutCallback restore;
        VecSim_SetTimeoutCallbackFunction(forceTimeout);
        timed_out_batch = batches(index.get(), query, eligible, 3, runtime);
    }
    check(timeout_callback_calls > 0, "The forced VecSim timeout callback was not called");
    check(timed_out_batch.batches == 1, "Expected a timeout from the first actual VecSim batch");
    check(timed_out_batch.status == "timeout" && timed_out_batch.neighbors.empty(),
          "A VecSim batch timeout must return timeout with no results");

    runtime.deadline = Clock::now() + std::chrono::hours(1);
    checkRecovered(scan(index.get(), query, eligible, 3, runtime), "NATIVE_SCAN");
    checkRecovered(batches(index.get(), query, eligible, 3, runtime), "NATIVE_BATCHES");
}

} // namespace

int main() {
    try {
        VecSim_UpdateThreadPoolSize(0);
#if !HAVE_SVS
        check(VecSim_GetSharedMemory() == 0, "SVS-disabled builds must have no shared SVS memory");
        VecSim_UpdateThreadPoolSize(1);
        check(VecSim_GetSharedMemory() == 0, "Resizing an absent SVS pool must allocate no memory");
        VecSim_UpdateThreadPoolSize(0);
#endif
        testTimeoutsAndRecovery();
        std::cout << "Native adapter timeout and recovery tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Native adapter test failure: " << error.what() << '\n';
        return 1;
    }
}
