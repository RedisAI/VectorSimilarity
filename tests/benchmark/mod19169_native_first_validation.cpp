/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3.
 */
#include "benchmark/benchmark.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/types/float16.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#ifndef BUILD_TESTS
#error "MOD19169 validation requires BUILD_TESTS and VectorSimilaritySerializer"
#endif
#ifndef MOD19169_NATIVE_FP16
#error "MOD19169 validation requires an explicit MOD19169_NATIVE_FP16 mode"
#endif

namespace {
static_assert(MOD19169_NATIVE_FP16 == 0 || MOD19169_NATIVE_FP16 == 1);
constexpr size_t query_count = 1000;
constexpr size_t k = 100;
constexpr size_t training_count = 10240;
constexpr size_t subset_count = 100000;
constexpr uint64_t cache_magic = 0x4d31393136395232ULL;
constexpr uint64_t graph_magic = 0x4d31393136394732ULL;

void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Hash {
    uint64_t value = 14695981039346656037ULL;
    void bytes(const void *data, size_t size) {
        const auto *p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < size; ++i) {
            value ^= p[i];
            value *= 1099511628211ULL;
        }
    }
    void word(uint64_t v) { bytes(&v, sizeof(v)); }
};

uint64_t file_hash(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "Cannot read " + path);
    Hash hash;
    std::array<char, 65536> buffer;
    while (input.read(buffer.data(), buffer.size()) || input.gcount())
        hash.bytes(buffer.data(), input.gcount());
    require(input.eof() && !input.bad(), "Failed reading " + path);
    return hash.value;
}

struct Options {
    std::string stage, dtype, dataset, corpus, source, queries, cache, graph, output;
    bool multi() const { return dataset == "multi"; }
    bool full() const { return corpus == "full"; }
    size_t dim() const { return multi() ? 512 : 768; }
    size_t source_count() const { return multi() ? 1111025 : 1000000; }
    size_t source_labels() const { return multi() ? 44441 : 1000000; }
};

Options parse(int &argc, char **argv) {
    Options o;
    std::map<std::string, std::string *> arguments = {
        {"--stage", &o.stage},   {"--dtype", &o.dtype},   {"--dataset", &o.dataset},
        {"--corpus", &o.corpus}, {"--source", &o.source}, {"--queries", &o.queries},
        {"--cache", &o.cache},   {"--graph", &o.graph},   {"--output", &o.output}};
    int remaining = 1;
    for (int i = 1; i < argc; ++i) {
        auto found = arguments.find(argv[i]);
        if (found == arguments.end()) {
            argv[remaining++] = argv[i];
        } else {
            require(i + 1 < argc, "Missing value for " + std::string(argv[i]));
            require(found->second->empty(), "Duplicate option " + std::string(argv[i]));
            *found->second = argv[++i];
        }
    }
    argc = remaining;
    argv[argc] = nullptr;
    require(o.stage == "query" || argc == 1, "Unrecognized reference/build argument");
    require(o.stage == "reference" || o.stage == "build" || o.stage == "query",
            "--stage must be reference, build, or query");
    require(o.dtype == "fp16" || o.dtype == "fp32", "--dtype must be fp16 or fp32");
    require(o.dataset == "single" || o.dataset == "multi", "--dataset must be single or multi");
    require(o.corpus == "subset" || o.corpus == "full", "--corpus must be subset or full");
    require(o.stage != "build" || !o.full(), "Build is only supported for subset corpora");
    require(!o.source.empty() && !o.queries.empty() && !o.cache.empty() && !o.output.empty(),
            "--source, --queries, --cache, and --output are required");
    require(o.stage == "reference" || !o.graph.empty(), "--graph is required for build/query");
    return o;
}

struct IndexDeleter {
    void operator()(VecSimIndex *index) const { VecSimIndex_Free(index); }
};
using Index = std::unique_ptr<VecSimIndex, IndexDeleter>;
struct ReplyDeleter {
    void operator()(VecSimQueryReply *reply) const { VecSimQueryReply_Free(reply); }
};
using Reply = std::unique_ptr<VecSimQueryReply, ReplyDeleter>;

// Cache identity is arithmetic-mode independent so both binaries share one exhaustive reference.
using Identity = std::array<uint64_t, 13>;
struct Reference {
    std::vector<uint64_t> ids;
    uint64_t ties = 0;
    uint64_t checksum = 0;
};

uint64_t reference_checksum(const Reference &reference) {
    Hash hash;
    hash.word(reference.ties);
    hash.bytes(reference.ids.data(), reference.ids.size() * sizeof(uint64_t));
    return hash.value;
}

std::vector<std::pair<uint64_t, double>> results(VecSimQueryReply *reply, size_t width,
                                                 const std::unordered_set<uint64_t> &labels) {
    require(reply && VecSimQueryReply_GetCode(reply) == VecSim_QueryReply_OK &&
                VecSimQueryReply_Len(reply) == width,
            "Query did not return the required successful width " + std::to_string(width));
    std::unique_ptr<VecSimQueryReply_Iterator, decltype(&VecSimQueryReply_IteratorFree)> iterator(
        VecSimQueryReply_GetIterator(reply), VecSimQueryReply_IteratorFree);
    std::vector<std::pair<uint64_t, double>> row;
    std::unordered_set<uint64_t> unique;
    while (VecSimQueryReply_IteratorHasNext(iterator.get())) {
        const auto *item = VecSimQueryReply_IteratorNext(iterator.get());
        uint64_t id = VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        require(std::isfinite(score) && unique.insert(id).second && labels.contains(id),
                "Query returned a nonfinite score, duplicate label, or unknown label");
        require(row.empty() || row.back().second <= score, "Query scores are not sorted");
        row.emplace_back(id, score);
    }
    require(row.size() == width, "Query iterator width differs from reply width");
    return row;
}

template <typename T>
float widen(T value) {
    if constexpr (std::is_same_v<T, vecsim_types::float16>)
        return vecsim_types::FP16_to_FP32(value);
    else
        return value;
}

template <typename T>
struct Inputs {
    Index source;
    HNSWIndex<T, float> *hnsw;
    std::vector<size_t> selected;
    std::unordered_set<uint64_t> labels;
    std::vector<T> queries;
    std::vector<float> mean;
    Identity identity{};
    uint64_t mean_hash = 0;

    explicit Inputs(const Options &o) : source(HNSWFactory::NewIndex(o.source)) {
        hnsw = dynamic_cast<HNSWIndex<T, float> *>(source.get());
        require(hnsw && hnsw->getDim() == o.dim() && hnsw->isMultiValue() == o.multi() &&
                    hnsw->getStoredDataSize() == o.dim() * sizeof(T) &&
                    hnsw->indexSize() == o.source_count() &&
                    hnsw->indexLabelCount() == o.source_labels(),
                "Source is not the expected complete uncompressed typed fixture");
        std::map<uint64_t, size_t> label_counts;
        for (size_t id = 0; id < hnsw->indexSize(); ++id)
            ++label_counts[hnsw->getExternalLabel(id)];
        require(label_counts.size() == o.source_labels(), "Source label metadata differs");
        if (o.multi()) {
            for (const auto &[label, count] : label_counts)
                require(count == 25, "Fashion source must contain 25 vectors per label");
        }
        if (o.full()) {
            for (const auto &[label, count] : label_counts)
                labels.insert(label);
        } else if (o.multi()) {
            std::vector<uint64_t> sorted_labels;
            for (const auto &[label, count] : label_counts)
                sorted_labels.push_back(label);
            for (size_t i = 0; i < subset_count / 25; ++i)
                labels.insert(sorted_labels[i * sorted_labels.size() / (subset_count / 25)]);
        }
        if (o.full() || o.multi()) {
            for (size_t id = 0; id < hnsw->indexSize(); ++id) {
                if (labels.contains(hnsw->getExternalLabel(id)))
                    selected.push_back(id);
            }
        } else {
            for (size_t i = 0; i < subset_count; ++i) {
                selected.push_back(i * hnsw->indexSize() / subset_count);
                labels.insert(hnsw->getExternalLabel(selected.back()));
            }
        }
        require(selected.size() == (o.full() ? o.source_count() : subset_count) &&
                    labels.size() > k,
                "Selection does not have the expected vector/label count");
        Hash selection_hash, data_hash;
        std::vector<double> sums(o.dim(), 0.0);
        for (size_t rank = 0; rank < selected.size(); ++rank) {
            size_t id = selected[rank];
            selection_hash.word(id);
            selection_hash.word(hnsw->getExternalLabel(id));
            const auto *stored = reinterpret_cast<const T *>(hnsw->getDataByInternalId(id));
            data_hash.bytes(stored, o.dim() * sizeof(T));
            for (size_t d = 0; d < o.dim(); ++d) {
                float value = widen(stored[d]);
                require(std::isfinite(value), "Source has a nonfinite value");
                if (rank < training_count)
                    sums[d] += value;
            }
        }
        mean.resize(o.dim());
        for (size_t d = 0; d < o.dim(); ++d)
            mean[d] = static_cast<float>(sums[d] / training_count);
        Hash mean_identity;
        mean_identity.bytes(mean.data(), mean.size() * sizeof(float));
        mean_hash = mean_identity.value;
        queries.resize(query_count * o.dim());
        std::ifstream input(o.queries, std::ios::binary);
        require(
            bool(input.read(reinterpret_cast<char *>(queries.data()), queries.size() * sizeof(T))),
            "Query raw file has fewer than 1000 typed rows");
        for (T value : queries)
            require(std::isfinite(widen(value)), "Raw query has a nonfinite value");
        for (size_t q = 0; q < query_count; ++q)
            VecSim_Normalize(queries.data() + q * o.dim(), o.dim(),
                             sizeof(T) == 2 ? VecSimType_FLOAT16 : VecSimType_FLOAT32);
        for (T value : queries)
            require(std::isfinite(widen(value)), "Normalized query has a nonfinite value");
        Hash query_hash;
        query_hash.bytes(queries.data(), queries.size() * sizeof(T));
        identity = {cache_magic,
                    2,
                    sizeof(T),
                    o.dim(),
                    uint64_t(o.multi()),
                    uint64_t(o.full()),
                    o.source_count(),
                    o.source_labels(),
                    selected.size(),
                    labels.size(),
                    selection_hash.value,
                    data_hash.value,
                    query_hash.value};
    }
};

template <typename T>
void validate_reference(const Reference &reference, const Inputs<T> &inputs) {
    require(reference.ids.size() == query_count * k && reference.ties <= query_count &&
                reference_checksum(reference) == reference.checksum,
            "Ground-truth cache payload is corrupt");
    for (size_t q = 0; q < query_count; ++q) {
        std::unordered_set<uint64_t> unique;
        for (size_t rank = 0; rank < k; ++rank) {
            uint64_t id = reference.ids[q * k + rank];
            require(inputs.labels.contains(id) && unique.insert(id).second,
                    "Ground-truth cache has an unknown or duplicate label");
        }
    }
}

template <typename T>
Reference load_reference(const Options &o, const Inputs<T> &inputs) {
    std::ifstream input(o.cache, std::ios::binary);
    Identity identity{};
    std::array<uint64_t, 4> metadata{};
    Reference reference;
    reference.ids.resize(query_count * k);
    require(bool(input.read(reinterpret_cast<char *>(identity.data()), sizeof(identity))) &&
                identity == inputs.identity &&
                bool(input.read(reinterpret_cast<char *>(metadata.data()), sizeof(metadata))) &&
                metadata[0] == query_count && metadata[1] == k &&
                bool(input.read(reinterpret_cast<char *>(reference.ids.data()),
                                reference.ids.size() * sizeof(uint64_t))) &&
                input.peek() == std::char_traits<char>::eof() && !input.bad(),
            "Ground-truth cache is stale, wrong, truncated, or absent: " + o.cache);
    reference.ties = metadata[2];
    reference.checksum = metadata[3];
    validate_reference(reference, inputs);
    return reference;
}

template <typename T>
Reference make_reference(const Options &o, const Inputs<T> &inputs) {
    require(!std::filesystem::exists(o.cache), "Reference requires a fresh cache path");
    BFParams params{};
    params.type = VecSimType_FLOAT32;
    params.dim = o.dim();
    params.metric = VecSimMetric_IP;
    params.multi = o.multi();
    params.blockSize = 1024;
    Index bf(BruteForceFactory::NewIndex(&params));
    require(bool(bf), "Cannot create exhaustive FP32 BF");
    std::vector<float> vector(o.dim());
    for (size_t id : inputs.selected) {
        const auto *stored = reinterpret_cast<const T *>(inputs.hnsw->getDataByInternalId(id));
        for (size_t d = 0; d < o.dim(); ++d)
            vector[d] = widen(stored[d]);
        VecSimIndex_AddVector(bf.get(), vector.data(), inputs.hnsw->getExternalLabel(id));
    }
    require(bf->indexSize() == inputs.selected.size() &&
                bf->indexLabelCount() == inputs.labels.size(),
            "Exhaustive FP32 BF differs from selected corpus");
    Reference reference;
    reference.ids.reserve(query_count * k);
    for (size_t q = 0; q < query_count; ++q) {
        for (size_t d = 0; d < o.dim(); ++d)
            vector[d] = widen(inputs.queries[q * o.dim() + d]);
        Reply reply(VecSimIndex_TopKQuery(bf.get(), vector.data(), k + 1, nullptr, BY_SCORE));
        auto row = results(reply.get(), k + 1, inputs.labels);
        for (size_t rank = 0; rank < k; ++rank)
            reference.ids.push_back(row[rank].first);
        reference.ties += row[k - 1].second == row[k].second;
    }
    reference.checksum = reference_checksum(reference);
    validate_reference(reference, inputs);
    std::array<uint64_t, 4> metadata{query_count, k, reference.ties, reference.checksum};
    std::ofstream output(o.cache, std::ios::binary);
    output.write(reinterpret_cast<const char *>(inputs.identity.data()), sizeof(inputs.identity));
    output.write(reinterpret_cast<const char *>(metadata.data()), sizeof(metadata));
    output.write(reinterpret_cast<const char *>(reference.ids.data()),
                 reference.ids.size() * sizeof(uint64_t));
    require(bool(output.flush()), "Cannot write ground-truth cache");
    return reference;
}

struct GraphIdentity {
    Identity inputs{};
    // magic, mean hash, stored byte checksum, serialized file checksum, builder mode, seed.
    std::array<uint64_t, 6> metadata{};
};

template <typename T>
uint64_t stored_checksum(HNSWIndex<T, float> *graph) {
    Hash hash;
    for (size_t id = 0; id < graph->indexSize(); ++id) {
        hash.word(graph->getExternalLabel(id));
        hash.bytes(graph->getDataByInternalId(id), graph->getStoredDataSize());
    }
    return hash.value;
}

template <typename T>
GraphIdentity build_graph(const Options &o, const Inputs<T> &inputs, double &seconds,
                          uintmax_t &size) {
    require(!std::filesystem::exists(o.graph) && !std::filesystem::exists(o.graph + ".identity"),
            "Build requires a fresh graph path");
    std::vector<float> initial_mean(o.dim(), 0.0f);
    HNSWParams params{};
    params.type = sizeof(T) == 2 ? VecSimType_FLOAT16 : VecSimType_FLOAT32;
    params.dim = o.dim();
    params.metric = VecSimMetric_IP;
    params.multi = o.multi();
    params.blockSize = 1024;
    params.M = 64;
    params.efConstruction = 512;
    params.efRuntime = 100;
    params.quantType = VecSimQuant_SQ8;
    params.quantParams = initial_mean.data();
    Index index(HNSWFactory::NewIndex(&params));
    auto *graph = dynamic_cast<HNSWIndex<T, float> *>(index.get());
    require(graph != nullptr, "Cannot create typed SQ8 graph");
    graph->setQuantizationMean(inputs.mean);
    const auto start = std::chrono::steady_clock::now();
    for (size_t id : inputs.selected)
        VecSimIndex_AddVector(index.get(), inputs.hnsw->getDataByInternalId(id),
                              inputs.hnsw->getExternalLabel(id));
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    require(graph->indexSize() == inputs.selected.size() &&
                graph->indexLabelCount() == inputs.labels.size(),
            "Built graph differs from selected corpus");
    GraphIdentity identity;
    identity.inputs = inputs.identity;
    identity.metadata = {graph_magic, inputs.mean_hash,   stored_checksum(graph),
                         0,           MOD19169_NATIVE_FP16, 100};
    graph->saveIndex(o.graph);
    size = std::filesystem::file_size(o.graph);
    identity.metadata[3] = file_hash(o.graph);
    std::ofstream output(o.graph + ".identity", std::ios::binary);
    output.write(reinterpret_cast<const char *>(identity.inputs.data()), sizeof(identity.inputs));
    output.write(reinterpret_cast<const char *>(identity.metadata.data()),
                 sizeof(identity.metadata));
    require(bool(output.flush()), "Cannot write graph identity");
    return identity;
}

std::string json_string(const std::string &s) {
    std::string result = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') {
            result += '\\';
            result += c;
        } else {
            require(c >= 32, "Control character in report path");
            result += c;
        }
    }
    return result + "\"";
}

template <typename T>
void report(const Options &o, const Inputs<T> &inputs, const Reference *reference,
            const GraphIdentity *graph, double seconds = 0, uintmax_t size = 0,
            const std::map<size_t, std::vector<size_t>> *hits = nullptr) {
    std::ofstream output(o.output);
    output << std::setprecision(17);
    output << "{\n  \"stage\": " << json_string(o.stage)
           << ",\n  \"dtype\": " << json_string(o.dtype)
           << ",\n  \"dataset\": " << json_string(o.dataset)
           << ",\n  \"corpus\": " << json_string(o.corpus)
           << ",\n  \"query_mode\": " << MOD19169_NATIVE_FP16
           << ",\n  \"source\": " << json_string(o.source)
           << ",\n  \"queries\": " << json_string(o.queries)
           << ",\n  \"graph\": " << json_string(o.graph) << ",\n  \"dimension\": " << o.dim()
           << ",\n  \"source_vectors\": " << o.source_count()
           << ",\n  \"source_labels\": " << o.source_labels()
           << ",\n  \"selected_vectors\": " << inputs.selected.size()
           << ",\n  \"selected_labels\": " << inputs.labels.size() << ",\n  \"selection_hash\": \""
           << inputs.identity[10] << "\",\n  \"selected_data_hash\": \"" << inputs.identity[11]
           << "\",\n  \"query_input_hash\": \"" << inputs.identity[12]
           << "\",\n  \"query_count\": " << query_count << ",\n  \"k\": " << k
           << ",\n  \"M\": 64,\n  \"ef_construction\": 512"
           << ",\n  \"normalization\": \"source already normalized; raw typed queries normalized "
              "exactly once\""
           << ",\n  \"training_vectors\": " << (o.full() ? "null" : std::to_string(training_count))
           << ",\n  \"mean_hash\": "
           << (o.full() ? "null" : json_string(std::to_string(inputs.mean_hash)))
           << ",\n  \"seed\": " << (o.full() ? "null" : "100");
    if (reference)
        output << ",\n  \"ground_truth_checksum\": \"" << reference->checksum
               << "\",\n  \"boundary_tied_queries\": " << reference->ties;
    if (graph)
        output << ",\n  \"graph_builder_mode\": "
               << (o.full() ? "null" : std::to_string(graph->metadata[4]))
               << ",\n  \"stored_byte_checksum\": \"" << graph->metadata[2]
               << "\",\n  \"graph_file_checksum\": \"" << graph->metadata[3] << "\"";
    if (o.stage == "build")
        output << ",\n  \"serial_insertion_seconds\": " << seconds;
    if (o.stage != "reference")
        output << ",\n  \"serialized_graph_bytes\": " << size;
    if (hits) {
        output << ",\n  \"per_query_hits\": {";
        bool first = true;
        for (const auto &[ef, values] : *hits) {
            if (!first)
                output << ',';
            first = false;
            output << "\n    \"ef" << ef << "\": [";
            for (size_t q = 0; q < values.size(); ++q) {
                if (q)
                    output << ',';
                output << values[q];
            }
            output << ']';
        }
        output << "\n  }";
    }
    output << "\n}\n";
    require(bool(output.flush()), "Cannot write report " + o.output);
}

template <typename T>
int run(const Options &o, int argc, char **argv) {
    Inputs<T> inputs(o);
    if (o.stage == "reference") {
        auto reference = make_reference(o, inputs);
        report(o, inputs, &reference, nullptr);
        return 0;
    }
    if (o.stage == "build") {
        double seconds = 0;
        uintmax_t size = 0;
        auto graph = build_graph(o, inputs, seconds, size);
        report(o, inputs, nullptr, &graph, seconds, size);
        return 0;
    }
    auto reference = load_reference(o, inputs);
    GraphIdentity graph_identity;
    if (!o.full()) {
        std::ifstream manifest(o.graph + ".identity", std::ios::binary);
        require(bool(manifest.read(reinterpret_cast<char *>(graph_identity.inputs.data()),
                                   sizeof(graph_identity.inputs))) &&
                    bool(manifest.read(reinterpret_cast<char *>(graph_identity.metadata.data()),
                                       sizeof(graph_identity.metadata))) &&
                    manifest.peek() == std::char_traits<char>::eof() && !manifest.bad() &&
                    graph_identity.inputs == inputs.identity &&
                    graph_identity.metadata[0] == graph_magic &&
                    graph_identity.metadata[1] == inputs.mean_hash &&
                    graph_identity.metadata[4] <= 1 && graph_identity.metadata[5] == 100 &&
                    graph_identity.metadata[3] == file_hash(o.graph),
                "Graph identity is stale, corrupt, wrong, or absent");
    }
    Index index(HNSWFactory::NewIndex(o.graph, true));
    auto *graph = dynamic_cast<HNSWIndex<T, float> *>(index.get());
    require(graph && graph->getDim() == o.dim() && graph->isMultiValue() == o.multi() &&
                graph->getStoredDataSize() ==
                    vecsim_types::sq8::storage_bytes_count<VecSimMetric_IP, true>(o.dim()) &&
                graph->indexSize() == inputs.selected.size() &&
                graph->indexLabelCount() == inputs.labels.size(),
            "Loaded graph is not the expected SQ8 corpus");
    auto info = VecSimIndex_DebugInfo(index.get());
    require(info.hnswInfo.M == 64 && info.hnswInfo.efConstruction == 512,
            "Loaded graph build parameters differ");
    const auto metric = info.commonInfo.basicInfo.metric;
    require(metric == VecSimMetric_IP || (o.full() && metric == VecSimMetric_Cosine),
            "Loaded graph does not use IP distance over normalized vectors");
    if (o.full()) {
        std::map<uint64_t, size_t> graph_labels;
        for (size_t id = 0; id < graph->indexSize(); ++id)
            ++graph_labels[graph->getExternalLabel(id)];
        require(graph_labels.size() == inputs.labels.size(), "Saved graph label count differs");
        for (const auto &[label, count] : graph_labels)
            require(inputs.labels.contains(label) && count == (o.multi() ? 25 : 1),
                    "Saved graph label membership or vector multiplicity differs");
    } else {
        for (size_t id = 0; id < inputs.selected.size(); ++id)
            require(graph->getExternalLabel(id) ==
                        inputs.hnsw->getExternalLabel(inputs.selected[id]),
                    "Loaded graph label insertion order differs from selected corpus");
    }
    uint64_t stored_hash = stored_checksum(graph);
    if (!o.full())
        require(stored_hash == graph_identity.metadata[2],
                "Stored graph bytes differ from build identity");
    else {
        graph_identity.metadata = {graph_magic, 0, stored_hash, file_hash(o.graph), 0, 0};
    }
    report(o, inputs, &reference, &graph_identity, 0, std::filesystem::file_size(o.graph));
    // Release the large uncompressed fixture before timing the SQ8 graph.
    inputs.source.reset();
    inputs.hnsw = nullptr;
    bool failed = false;
    std::map<size_t, std::vector<size_t>> per_query_hits;
    for (size_t ef : {100UL, 200UL, 400UL}) {
        const std::string name =
            "MOD19169/" + o.dtype + "/" + o.dataset + "/" + o.corpus + "/ef" + std::to_string(ef);
        benchmark::RegisterBenchmark(
            name.c_str(),
            [&, ef](benchmark::State &state) {
                VecSimQueryParams params{};
                params.hnswRuntimeParams.efRuntime = ef;
                size_t q = 0;
                size_t correct = 0;
                std::vector<size_t> hits;
                hits.reserve(query_count);
                const T *query = inputs.queries.data();
                for (auto _ : state) {
                    auto *reply = VecSimIndex_TopKQuery(index.get(), query, k, &params, BY_SCORE);
                    state.PauseTiming();
                    try {
                        Reply owner(reply);
                        auto row = results(reply, k, inputs.labels);
                        auto first = reference.ids.begin() + q * k;
                        size_t query_hits = 0;
                        for (const auto &[id, score] : row)
                            query_hits += std::find(first, first + k, id) != first + k;
                        correct += query_hits;
                        hits.push_back(query_hits);
                    } catch (const std::exception &error) {
                        failed = true;
                        state.ResumeTiming();
                        state.SkipWithError(error.what());
                        return;
                    }
                    ++q;
                    query = inputs.queries.data() + (q % query_count) * o.dim();
                    state.ResumeTiming();
                }
                if (q != query_count ||
                    (per_query_hits.contains(ef) && per_query_hits.at(ef) != hits)) {
                    failed = true;
                    state.SkipWithError(
                        "Benchmark must execute the same 1000 query results in every repetition");
                    return;
                }
                per_query_hits[ef] = std::move(hits);
                state.counters["Recall_vs_FP32_BF"] = double(correct) / (k * q);
                state.counters["FP32_BF_boundary_tied_queries"] = reference.ties;
                state.counters["query_mode"] = MOD19169_NATIVE_FP16;
                state.SetLabel(
                    "reference=FP32_BF_over_original_typed_values; query includes preprocessing");
            })
            ->Iterations(query_count)
            ->Repetitions(3)
            ->Unit(benchmark::kMillisecond);
    }
    benchmark::Initialize(&argc, argv);
    require(!benchmark::ReportUnrecognizedArguments(argc, argv),
            "Unrecognized benchmark arguments");
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    report(o, inputs, &reference, &graph_identity, 0, std::filesystem::file_size(o.graph),
           &per_query_hits);
    return failed ? 1 : 0;
}

int numeric_domain_probe() {
    constexpr size_t dim = 2;
    const std::array<float, dim> storage{1e-37f, 1e-37f};
    const float max = std::numeric_limits<float>::max();
    const std::array<float, dim> query{max, -max};
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto components = CreateSQ8IndexComponents<float, VecSimMetric_IP>(allocator, dim, nullptr);
    std::unique_ptr<IndexCalculatorInterface<float>> calculator(components.indexCalculator);
    std::unique_ptr<PreprocessorsContainerAbstract> preprocessors(components.preprocessors);
    require(calculator && preprocessors, "Numeric-domain probe components are unavailable");
    auto storage_blob = preprocessors->preprocessForStorage(storage.data(), sizeof(storage));
    auto query_blob = preprocessors->preprocessQuery(query.data(), sizeof(query));
    require(storage_blob && query_blob, "Numeric-domain probe preprocessing failed");
    const auto dispatch = calculator->getDistanceDispatch(DistanceMode::StoredToQuery);
    require(dispatch.isValid(), "Numeric-domain probe query dispatch is unavailable");
    const float scalar =
        calculator->calcDistanceForQuery(storage_blob.get(), query_blob.get(), dim);
    const float cached = dispatch(storage_blob.get(), query_blob.get(), dim);
    const bool scalar_finite = std::isfinite(scalar);
    const bool cached_finite = std::isfinite(cached);
    const char *classification;
    if (scalar_finite != cached_finite || (scalar_finite && scalar != cached))
        classification = "dispatch_mismatch";
    else if (scalar_finite)
        classification = scalar == 1.0f ? "finite_expected_distance" : "finite_unexpected_distance";
    else
        classification =
            std::isnan(scalar) && std::isnan(cached) ? "nan_distance" : "nonfinite_distance";
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "{\n  \"query_mode\": " << MOD19169_NATIVE_FP16 << ",\n  \"scalar_score\": ";
    if (scalar_finite)
        std::cout << scalar;
    else
        std::cout << "null";
    std::cout << ",\n  \"scalar_finite\": " << (scalar_finite ? "true" : "false")
              << ",\n  \"cached_score\": ";
    if (cached_finite)
        std::cout << cached;
    else
        std::cout << "null";
    std::cout << ",\n  \"cached_finite\": " << (cached_finite ? "true" : "false")
              << ",\n  \"classification\": \"" << classification << "\"\n}\n";
    require(bool(std::cout.flush()), "Cannot write numeric-domain probe results");
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--numeric-domain-probe")
            return numeric_domain_probe();
        auto options = parse(argc, argv);
        if (options.dtype == "fp16")
            return run<vecsim_types::float16>(options, argc, argv);
        return run<float>(options, argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "MOD19169 validation: " << error.what() << '\n';
        return 1;
    }
}
