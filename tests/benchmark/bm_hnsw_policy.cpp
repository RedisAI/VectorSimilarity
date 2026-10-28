/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "VecSim/vec_sim.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using Hit = std::pair<double, size_t>;
using Heap = std::priority_queue<Hit>;
using Clock = std::chrono::steady_clock;

static void retain(Heap &heap, size_t k, Hit hit) {
    if (heap.size() < k) {
        heap.push(hit);
    } else if (hit < heap.top()) {
        heap.pop();
        heap.push(hit);
    }
}

static std::unordered_set<size_t> labels(Heap heap) {
    std::unordered_set<size_t> result;
    while (!heap.empty()) {
        result.insert(heap.top().second);
        heap.pop();
    }
    return result;
}

template <typename T>
static void measure(size_t n, size_t dim, size_t m, unsigned seed, bool clustered) {
    constexpr auto type = std::is_same_v<T, float> ? VecSimType_FLOAT32 : VecSimType_FLOAT64;
    VecSimParams params{};
    params.algo = VecSimAlgo_HNSWLIB;
    params.algoParams.hnswParams = {.type = type,
                                    .dim = dim,
                                    .metric = VecSimMetric_Cosine,
                                    .M = m,
                                    .efConstruction = 100,
                                    .efRuntime = 10};
    std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(VecSimIndex_New(&params),
                                                                    VecSimIndex_Free);
    if (!index) {
        throw std::runtime_error("could not create HNSW index");
    }
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal;
    std::vector<std::vector<T>> centers(16, std::vector<T>(dim));
    for (auto &center : centers) {
        for (auto &value : center) {
            value = normal(rng);
        }
    }
    auto vector = [&](size_t cluster) {
        std::vector<T> result(dim);
        for (size_t j = 0; j < dim; ++j) {
            result[j] =
                clustered ? centers[cluster % centers.size()][j] + 0.2 * normal(rng) : normal(rng);
        }
        return result;
    };
    for (size_t i = 0; i < n; ++i) {
        auto v = vector(i);
        VecSimIndex_AddVector(index.get(), v.data(), i);
    }
    std::cerr << "built n=" << n << " dim=" << dim << " M=" << m << " seed=" << seed
              << " bytes=" << sizeof(T) << " clustered=" << clustered << '\n';
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::cout << "n,dim,m,k,subset,bytes,seed,clustered,query,repeat,adhoc_us,batches_us,recall,"
                 "old_adhoc,batch_count\n";
    for (double ratio : {0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0}) {
        size_t subset = std::max(size_t(1), size_t(n * ratio));
        std::vector<size_t> selected(order.begin(), order.begin() + subset);
        std::sort(selected.begin(), selected.end());
        for (size_t k : {5, 12, 13, 50, 200, 500}) {
            for (size_t q = 0; q < 3; ++q) {
                auto query = vector(q);
                auto normalized = query;
                VecSim_Normalize(normalized.data(), dim, type);
                auto adhoc = [&]() {
                    Heap heap;
                    for (auto id : selected) {
                        retain(
                            heap, k,
                            {VecSimIndex_GetDistanceFrom_Unsafe(index.get(), id, normalized.data()),
                             id});
                    }
                    return heap;
                };
                auto batches = [&]() {
                    std::unique_ptr<VecSimBatchIterator, decltype(&VecSimBatchIterator_Free)> it(
                        VecSimBatchIterator_New(index.get(), query.data(), nullptr),
                        VecSimBatchIterator_Free);
                    Heap heap;
                    size_t estimate = subset;
                    while (heap.size() < k && VecSimBatchIterator_HasNext(it.get())) {
                        size_t remaining = k - heap.size();
                        size_t batch_size = std::min(
                            n, size_t(double(remaining) * n / std::max(size_t(1), estimate)) + 1);
                        std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)> reply(
                            VecSimBatchIterator_Next(it.get(), batch_size, BY_ID),
                            VecSimQueryReply_Free);
                        if (VecSimQueryReply_GetCode(reply.get()) != VecSim_QueryReply_OK) {
                            throw std::runtime_error("batch search did not complete");
                        }
                        auto before = heap.size();
                        std::unique_ptr<VecSimQueryReply_Iterator,
                                        decltype(&VecSimQueryReply_IteratorFree)>
                            results(VecSimQueryReply_GetIterator(reply.get()),
                                    VecSimQueryReply_IteratorFree);
                        auto filter = selected.begin();
                        while (VecSimQueryReply_IteratorHasNext(results.get())) {
                            auto hit = VecSimQueryReply_IteratorNext(results.get());
                            size_t id = VecSimQueryResult_GetId(hit);
                            filter = std::lower_bound(filter, selected.end(), id);
                            if (filter != selected.end() && *filter == id) {
                                retain(heap, k, {VecSimQueryResult_GetScore(hit), id});
                            }
                        }
                        // Same hit-rate update as RediSearch's forced-batches path.
                        estimate = std::min(
                            estimate,
                            (estimate + size_t(double(heap.size() - before) / remaining * n)) / 2);
                    }
                    return heap;
                };
                adhoc();
                batches();
                for (size_t repeat = 0; repeat < 3; ++repeat) {
                    Heap exact, approximate;
                    double adhoc_us, batches_us;
                    auto timed = [&](auto &run, Heap &result, double &us) {
                        auto start = Clock::now();
                        result = run();
                        us =
                            std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                    };
                    if ((q + repeat) % 2) {
                        timed(adhoc, exact, adhoc_us);
                        timed(batches, approximate, batches_us);
                    } else {
                        timed(batches, approximate, batches_us);
                        timed(adhoc, exact, adhoc_us);
                    }
                    if (exact.size() != std::min(k, subset)) {
                        throw std::runtime_error("exact search returned the wrong result count");
                    }
                    auto truth = labels(exact);
                    auto found = labels(approximate);
                    size_t overlap = 0;
                    for (auto id : found) {
                        overlap += truth.contains(id);
                    }
                    std::cout << n << ',' << dim << ',' << m << ',' << k << ',' << subset << ','
                              << sizeof(T) << ',' << seed << ',' << clustered << ',' << q << ','
                              << repeat << ',' << std::setprecision(10) << adhoc_us << ','
                              << batches_us << ',' << double(overlap) / truth.size() << ','
                              << VecSimIndex_PreferAdHocSearch(index.get(), subset, k, true) << ','
                              << approximate.size() << '\n';
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 7) {
            throw std::invalid_argument("usage: bm_hnsw_policy n dim M seed bytes clustered(0|1)");
        }
        size_t n = std::stoull(argv[1]), dim = std::stoull(argv[2]), m = std::stoull(argv[3]);
        auto seed = std::stoul(argv[4]);
        auto bytes = std::stoul(argv[5]);
        auto clustered = std::stoul(argv[6]);
        if (!n || !dim || m < 2 || (bytes != 4 && bytes != 8) || clustered > 1) {
            throw std::invalid_argument("invalid benchmark parameters");
        }
        if (bytes == 4) {
            measure<float>(n, dim, m, seed, clustered);
        } else {
            measure<double>(n, dim, m, seed, clustered);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
