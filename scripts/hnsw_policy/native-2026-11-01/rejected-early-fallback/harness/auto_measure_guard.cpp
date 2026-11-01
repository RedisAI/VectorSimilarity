/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/vec_sim.h"
#include "../progress-policy.h"
#include "../guard-policy.h"
#define preferHnswAdhocMeasured originalDraftPolicy
#include "../original-draft-policy.h"
#undef preferHnswAdhocMeasured
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Hit = std::pair<double, size_t>;
using Heap = std::priority_queue<Hit>;
using Index = std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)>;
static double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t).count();
}
static void retain(Heap &heap, size_t k, Hit hit) {
    if (heap.size() < k) heap.push(hit);
    else if (hit < heap.top()) { heap.pop(); heap.push(hit); }
}
static std::set<size_t> ids(Heap heap) {
    std::set<size_t> result;
    while (!heap.empty()) { result.insert(heap.top().second); heap.pop(); }
    return result;
}
static void read_floats(std::ifstream &in, std::vector<float> &v) {
    if (!in.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float)))
        throw std::runtime_error("short vector input");
}
static void extract(const char *source, const char *output, size_t n, size_t seed) {
    Index index(HNSWFactory::NewIndex(source), VecSimIndex_Free);
    auto *hnsw = dynamic_cast<HNSWIndex<float, float> *>(index.get());
    if (!hnsw || n > index->indexSize()) throw std::runtime_error("bad extract source/count");
    std::vector<size_t> order(index->indexSize());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::ofstream out(output, std::ios::binary);
    for (size_t i = 0; i < n; ++i)
        out.write(hnsw->getDataByInternalId(order[i]), hnsw->getDim() * sizeof(float));
    if (!out) throw std::runtime_error("extract write failed");
}

template <typename T>
static void measure(const char *vectors, const char *queries, size_t n, size_t dim, size_t m,
                    size_t efc, size_t qoffset, size_t nq, size_t seed, const char *trace_path, const std::string &suite) {
    constexpr auto type = std::is_same_v<T, float> ? VecSimType_FLOAT32 : VecSimType_FLOAT64;
    VecSimParams params{};
    params.algo = VecSimAlgo_HNSWLIB;
    params.algoParams.hnswParams = {.type = type, .dim = dim, .metric = VecSimMetric_Cosine,
                                    .M = m, .efConstruction = efc, .efRuntime = 10};
    const char *cache = std::getenv("MOD18264_GRAPH_CACHE");
    bool cached = cache && std::filesystem::exists(cache);
    Index index(cached ? HNSWFactory::NewIndex(cache) : VecSimIndex_New(&params), VecSimIndex_Free);
    auto *typed_index = dynamic_cast<HNSWIndex<T, T> *>(index.get());
    if (!typed_index || (cached && (index->indexSize() != n || typed_index->getDim() != dim || typed_index->getType() != type || typed_index->getM() != m || typed_index->getEfConstruction() != efc)))
        throw std::runtime_error("cached graph shape differs");
    if (!index) throw std::runtime_error("index allocation failed");
    std::ifstream data(vectors, std::ios::binary), query_file(queries, std::ios::binary);
    std::vector<float> raw(dim);
    std::vector<T> typed(dim);
    std::vector<float> projection(n);
    for (size_t i = 0; i < n; ++i) {
        read_floats(data, raw);
        projection[i] = raw[0];
        std::copy(raw.begin(), raw.end(), typed.begin());
        if (!cached) VecSimIndex_AddVector(index.get(), typed.data(), i);
    }
    if (cache && !cached) {
        auto *hnsw = dynamic_cast<HNSWIndex<T, T> *>(index.get());
        if (!hnsw) throw std::runtime_error("cache index type differs");
        hnsw->saveIndex(cache);
    }
    std::cerr << (cached ? "loaded n=" : "built n=") << n << " dim=" << dim << " M=" << m << " efc=" << efc
              << " bytes=" << sizeof(T) << '\n';
    std::vector<std::vector<T>> query_vectors;
    query_file.seekg(qoffset * dim * sizeof(float));
    for (size_t q = 0; q < nq; ++q) {
        read_floats(query_file, raw);
        query_vectors.emplace_back(raw.begin(), raw.end());
    }
    std::ofstream trace(trace_path);
    if (!trace) throw std::runtime_error("trace output open failed");
    trace << "n,dim,m,efc,bytes,seed,filter,ratio,k,query,estimate_kind,ef,repeat,policy,batch,requested,returned,remaining_before,new_hits,estimate_before,estimate_after,cumulative_us\n";
    std::cout << "n,dim,m,efc,bytes,seed,filter,ratio,k,query,estimate_kind,ef,repeat,policy,adhoc_us,batches_us,recall,returned,batch_count,tie_adjusted_recall,subset,switched\n";
    std::vector<size_t> order(n);
    std::mt19937 rng(seed);
    const bool quality_suite = suite.ends_with(":quality");
    const bool nearest_edges = suite.ends_with(":nearestedges");
    const bool edge_suite = suite.ends_with(":edges") || nearest_edges;
    std::vector<std::pair<double, size_t>> subsets;
    if (edge_suite) {
        for (size_t count : {199, 200, 201}) subsets.emplace_back(double(count)/n, count);
    } else if (quality_suite) {
        subsets.emplace_back(.26, size_t(n * .26));
    } else {
        for (double ratio : {0.01, 0.05, 0.10, 0.175, 0.26, 0.5, 1.0})
            subsets.emplace_back(ratio, std::max(size_t(1), size_t(n * ratio)));
    }
    const std::vector<size_t> k_values = (edge_suite || quality_suite) ? std::vector<size_t>{200} : std::vector<size_t>{13,50,200,500};
    const std::vector<size_t> ef_values = quality_suite ? std::vector<size_t>{10,200,400,800} : std::vector<size_t>{10};
    const std::vector<std::string> filters = nearest_edges ? std::vector<std::string>{"nearest"} : std::vector<std::string>{"random", "projection"};
    for (const std::string filter : filters) {
        std::iota(order.begin(), order.end(), 0);
        if (filter == "random") std::shuffle(order.begin(), order.end(), rng);
        else std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return projection[a] < projection[b]; });
        for (auto [ratio, subset] : subsets) {
            std::vector<size_t> selected(order.begin(), order.begin() + subset);
            std::sort(selected.begin(), selected.end());
            for (size_t k : k_values) {
                for (size_t q = 0; q < nq; ++q) {
                    auto normalized = query_vectors[q];
                    VecSim_Normalize(normalized.data(), dim, type);
                    if (filter == "nearest") {
                        Heap nearest;
                        for (size_t id = 0; id < n; ++id)
                            retain(nearest, subset, {VecSimIndex_GetDistanceFrom_Unsafe(index.get(), id, normalized.data()), id});
                        auto nearest_ids = ids(nearest);
                        selected.assign(nearest_ids.begin(), nearest_ids.end());
                    }
                    auto adhoc = [&]() {
                        Heap heap;
                        for (auto id : selected)
                            retain(heap, k, {VecSimIndex_GetDistanceFrom_Unsafe(index.get(), id, normalized.data()), id});
                        return heap;
                    };
                    auto exact_heap = adhoc();
                    double exact_cutoff = exact_heap.top().first;
                    auto truth = ids(exact_heap);
                    if (truth.size() != std::min(k, subset)) throw std::runtime_error("wrong reference count");
                    for (const std::string estimate_kind : {"actual", "full"}) {
                        for (size_t ef : ef_values) {
                            for (size_t repeat = 0; repeat < 3; ++repeat) {
                              std::optional<std::set<size_t>> progress_result;
                              std::optional<std::set<size_t>> control_ids, guarded_ids;
                              bool guard_switched = false;
                              std::vector<std::string> policies{"forced", "draft", "progress", "compiled"};
                              if (suite.find(":guard") != std::string::npos) policies = {"compiled", "guarded"};
                              std::rotate(policies.begin(), policies.begin() + (q + repeat) % policies.size(), policies.end());
                              for (const auto &policy : policies) {
                                double bf_us = 0;
                                auto timed_bf = [&]() { auto start = Clock::now(); auto hits = adhoc(); bf_us = elapsed(start); return hits.size(); };
                                if ((q + repeat) % 2 == 0 && timed_bf() != truth.size()) throw std::runtime_error("BF count changed");
                                VecSimQueryParams qp{};
                                qp.hnswRuntimeParams.efRuntime = ef;
                                auto start = Clock::now();
                                VecSimBatchIterator *it = nullptr;
                                Heap heap;
                                size_t estimate = estimate_kind == "actual" ? subset : n;
                                size_t batches = 0;
                                bool switched = false;
                                auto prefer = [&](size_t remaining, bool initial) {
                                    const float r = float(estimate) / float(n);
                                    if (policy == "forced") return false;
                                    if (policy == "guarded") return guardPolicy(k, n, dim, m, r, initial ? 0 : remaining);
                                    if (policy == "draft") return originalDraftPolicy(k, n, dim, m, r);
                                    if (policy == "progress") return progressPolicy(k, n, dim, m, r, initial ? 0 : remaining);
                                    return initial ? bool(VecSimIndex_PreferAdHocSearch(index.get(), estimate, k, true))
                                                   : bool(VecSimIndex_PreferAdHocSearchAfterBatch(index.get(), estimate, k, remaining));
                                };
                                if (prefer(k, true)) heap = adhoc();
                                else it = VecSimBatchIterator_New(index.get(), query_vectors[q].data(), &qp);
                                struct Step { size_t requested, returned, remaining, hits, old_est, new_est; double us; };
                                std::vector<Step> steps;
                                while (it && heap.size() < k && VecSimBatchIterator_HasNext(it)) {
                                    size_t remaining = k - heap.size();
                                    size_t batch_size = size_t(double(remaining) * n / std::max(size_t(1), estimate)) + 1;
                                    auto *reply = VecSimBatchIterator_Next(it, batch_size, BY_ID);
                                    if (VecSimQueryReply_GetCode(reply) != VecSim_QueryReply_OK) throw std::runtime_error("batch failed");
                                    size_t returned = VecSimQueryReply_Len(reply), before = heap.size();
                                    auto *results = VecSimQueryReply_GetIterator(reply);
                                    auto filter_it = selected.begin();
                                    while (VecSimQueryReply_IteratorHasNext(results)) {
                                        auto *hit = VecSimQueryReply_IteratorNext(results);
                                        size_t id = VecSimQueryResult_GetId(hit);
                                        filter_it = std::lower_bound(filter_it, selected.end(), id);
                                        if (filter_it != selected.end() && *filter_it == id)
                                            retain(heap, k, {VecSimQueryResult_GetScore(hit), id});
                                    }
                                    VecSimQueryReply_IteratorFree(results);
                                    VecSimQueryReply_Free(reply);
                                    size_t old_est = estimate;
                                    estimate = std::min(estimate, (estimate + size_t(double(heap.size() - before) / remaining * n)) / 2);
                                    steps.push_back({batch_size, returned, remaining, heap.size() - before, old_est, estimate, elapsed(start)});
                                    if (heap.size() < k && prefer(k - heap.size(), false)) {
                                        heap = adhoc();
                                        switched = true;
                                        ++batches;
                                        break;
                                    }
                                    if (++batches > n + 1) throw std::runtime_error("batch search did not terminate");
                                }
                                if (it) VecSimBatchIterator_Free(it);
                                double batch_us = elapsed(start);
                                if ((q + repeat) % 2 != 0 && timed_bf() != truth.size()) throw std::runtime_error("BF count changed");
                                auto found = ids(heap);
                                if (policy == "compiled") control_ids = found;
                                if (policy == "guarded") {
                                    guarded_ids = found; guard_switched = switched || batches == 0;
                                    if ((switched || batches == 0) && found != truth) throw std::runtime_error("guard BF differs from ground truth");
                                }
                                if (policy == "progress" || policy == "compiled") {
                                    if (progress_result && *progress_result != found)
                                        throw std::runtime_error("compiled policy disagrees with prototype results");
                                    progress_result = found;
                                }
                                size_t overlap = 0;
                                size_t tie_hits = 0;
                                for (auto id : found) {
                                    overlap += truth.contains(id);
                                    tie_hits += VecSimIndex_GetDistanceFrom_Unsafe(index.get(), id, normalized.data()) <= exact_cutoff;
                                }
                                if (found.size() != truth.size()) throw std::runtime_error("incomplete batch result");
                                auto prefix = [&](std::ostream &out) -> std::ostream & {
                                    return out << std::setprecision(10) << n << ',' << dim << ',' << m << ',' << efc << ',' << sizeof(T) << ',' << seed << ','
                                               << filter << ',' << ratio << ',' << k << ',' << qoffset + q << ',' << estimate_kind << ',' << ef << ',' << repeat << ',' << policy << ',';
                                };
                                prefix(std::cout) << std::setprecision(10) << bf_us << ',' << batch_us << ',' << double(overlap) / truth.size() << ',' << found.size() << ',' << batches << ',' << double(tie_hits) / truth.size() << ',' << subset << ',' << switched << '\n';
                                for (size_t b = 0; b < steps.size(); ++b) {
                                    const auto &s = steps[b];
                                    prefix(trace) << b << ',' << s.requested << ',' << s.returned << ',' << s.remaining << ',' << s.hits << ','
                                                  << s.old_est << ',' << s.new_est << ',' << s.us << '\n';
                                }
                              }
                              if (control_ids && guarded_ids && !guard_switched && *control_ids != *guarded_ids)
                                  throw std::runtime_error("guard changed non-fallback results");
                            }
                        }
                    }
                }
            }
            std::cout.flush(); trace.flush();
        }
    }
}
int main(int argc, char **argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "extract") {
            extract(argv[2], argv[3], std::stoull(argv[4]), std::stoull(argv[5]));
        } else if (argc == 13) {
            auto run = [&](auto tag) {
                measure<decltype(tag)>(argv[1], argv[2], std::stoull(argv[3]), std::stoull(argv[4]), std::stoull(argv[5]),
                                      std::stoull(argv[6]), std::stoull(argv[8]), std::stoull(argv[9]), std::stoull(argv[10]), argv[11], argv[12]);
            };
            // A :edges suffix restricts the sweep to K-1, K, and K+1 matching documents.
            if (std::string(argv[7]) == "4") run(float{});
            else if (std::string(argv[7]) == "8") run(double{});
            else throw std::runtime_error("bytes must be 4 or 8");
        } else {
            throw std::runtime_error("usage: measure vectors.raw queries.raw n dim M efc bytes qoffset nq seed trace.csv dataset_label; or measure extract source.hnsw output.raw n seed");
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
