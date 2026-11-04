/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "native_candidates.h"
#include "VecSim/algorithms/hnsw/hnsw_serializer.h"
#include "VecSim/index_factories/hnsw_factory.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

using namespace filtered_bench;
namespace fs = std::filesystem;

namespace {

using Index = std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)>;

void logToStderr(void *, const char *level, const char *message) {
    std::cerr << "VecSim [" << (level ? level : "unknown") << "]: " << (message ? message : "")
              << '\n';
}

std::string jsonString(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            output << '\\' << ch;
        } else if (ch < 32) {
            output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(ch) << std::dec;
        } else {
            output << ch;
        }
    }
    output << '"';
    return output.str();
}

size_t unsignedValue(std::string_view text, bool allow_zero = false) {
    size_t value = 0;
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size() || (!allow_zero && value == 0)) {
        throw std::runtime_error("Invalid unsigned integer: " + std::string(text));
    }
    return value;
}

class Arguments {
    std::map<std::string, std::string> values;

public:
    Arguments(int argc, char **argv, const std::set<std::string> &allowed) {
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 == argc || !allowed.contains(argv[i]) ||
                !values.emplace(argv[i], argv[i + 1]).second) {
                throw std::runtime_error("Unknown, duplicate, or incomplete argument: " +
                                         std::string(argv[i]));
            }
        }
    }
    const std::string &get(const std::string &name) const {
        auto value = values.find(name);
        if (value == values.end()) {
            throw std::runtime_error("Missing argument: " + name);
        }
        return value->second;
    }
    size_t number(const std::string &name) const { return unsignedValue(get(name)); }
};

size_t product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) {
        throw std::runtime_error("Input size overflow");
    }
    return a * b;
}

template <typename T>
std::vector<T> readArray(const fs::path &path, size_t expected_count) {
    size_t bytes = product(expected_count, sizeof(T));
    if (fs::file_size(path) != bytes ||
        bytes > size_t(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("Wrong payload length: " + path.string());
    }
    std::vector<T> values(expected_count);
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ios::badbit | std::ios::failbit);
    if (bytes) {
        input.read(reinterpret_cast<char *>(values.data()), bytes);
    }
    return values;
}

VecSimMetric parseMetric(const std::string &metric) {
    if (metric == "cosine") {
        return VecSimMetric_Cosine;
    }
    if (metric == "l2") {
        return VecSimMetric_L2;
    }
    throw std::runtime_error("Metric must be cosine or l2");
}

void validateVectors(std::span<const float> vectors, size_t dim, VecSimMetric metric) {
    for (size_t begin = 0; begin < vectors.size(); begin += dim) {
        double norm_squared = 0;
        for (size_t j = 0; j < dim; ++j) {
            float value = vectors[begin + j];
            if (!std::isfinite(value)) {
                throw std::runtime_error("Nonfinite vector component");
            }
            norm_squared += double(value) * value;
        }
        if (metric == VecSimMetric_Cosine && norm_squared == 0) {
            throw std::runtime_error("Zero-norm cosine vector");
        }
    }
}

double elapsedMs(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

void provenance(std::ostream &output) {
    output << "\"compiler\":" << jsonString(__VERSION__)
           << ",\"build_type\":" << jsonString(FILTERED_BUILD_TYPE)
           << ",\"source_revision\":" << jsonString(FILTERED_SOURCE_REVISION)
           << ",\"source_dirty\":" << FILTERED_SOURCE_DIRTY
           << ",\"shared_memory_bytes\":" << VecSim_GetSharedMemory()
           << ",\"write_mode\":\"in_place\"";
}

void validateIndex(VecSimIndex *index, size_t documents, size_t dimensions, VecSimMetric metric) {
    if (!index) {
        throw std::runtime_error("Failed to construct or load HNSW index");
    }
    auto info = VecSimIndex_BasicInfo(index);
    if (info.algo != VecSimAlgo_HNSWLIB || info.type != VecSimType_FLOAT32 || info.isMulti ||
        info.isTiered || info.isDisk || info.dim != dimensions || info.metric != metric ||
        VecSimIndex_IndexSize(index) != documents) {
        throw std::runtime_error(
            "Graph metadata does not match the requested FP32 single HNSW index");
    }
    auto *typed = dynamic_cast<VecSimIndexAbstract<float, float> *>(index);
    if (!typed || typed->usesQuantizedStorage()) {
        throw std::runtime_error("The native baselines require unquantized FP32 storage");
    }
}

void build(int argc, char **argv) {
    Arguments args(argc, argv,
                   {"--vectors", "--documents", "--dimensions", "--metric", "--graph", "--m",
                    "--ef-construction"});
    const size_t documents = args.number("--documents");
    if (documents >= std::numeric_limits<idType>::max()) {
        throw std::runtime_error("Document count exceeds the HNSW internal ID range");
    }
    const size_t dimensions = args.number("--dimensions");
    const auto metric = parseMetric(args.get("--metric"));
    const fs::path graph(args.get("--graph"));
    if (fs::exists(graph)) {
        throw std::runtime_error("Refusing to overwrite a graph");
    }
    const size_t m = args.number("--m");
    if (m < 2 || m > std::numeric_limits<uint16_t>::max() / 2) {
        throw std::runtime_error("M must be between 2 and 32767 (the HNSW link count limit)");
    }
    auto vectors = readArray<float>(args.get("--vectors"), product(documents, dimensions));
    validateVectors(vectors, dimensions, metric);
    VecSimParams params = {};
    params.algo = VecSimAlgo_HNSWLIB;
    auto &hnsw = params.algoParams.hnswParams;
    hnsw.type = VecSimType_FLOAT32;
    hnsw.dim = dimensions;
    hnsw.metric = metric;
    hnsw.M = m;
    hnsw.efConstruction = args.number("--ef-construction");
    hnsw.efRuntime = 10;
    auto begin = Clock::now();
    Index index(VecSimIndex_New(&params), &VecSimIndex_Free);
    if (!index) {
        throw std::runtime_error("Failed to create HNSW index");
    }
    for (size_t id = 0; id < documents; ++id) {
        if (VecSimIndex_AddVector(index.get(), vectors.data() + id * dimensions, id) != 1) {
            throw std::runtime_error("Failed to insert a new vector");
        }
    }
    double build_ms = elapsedMs(begin);
    validateIndex(index.get(), documents, dimensions, metric);
    auto *serializer = dynamic_cast<HNSWSerializer *>(index.get());
    if (!serializer) {
        throw std::runtime_error("HNSW serialization helpers are unavailable");
    }
    begin = Clock::now();
    serializer->saveIndex(graph.string());
    const double serialization_ms = elapsedMs(begin);
    if (!fs::is_regular_file(graph) || fs::file_size(graph) == 0) {
        throw std::runtime_error("Could not write graph snapshot");
    }
    std::cout << std::setprecision(17) << '{';
    provenance(std::cout);
    std::cout << ",\"build_ms\":" << build_ms << ",\"serialization_ms\":" << serialization_ms
              << ",\"index_bytes\":" << VecSimIndex_StatsInfo(index.get()).memory << "}\n";
}

struct Case {
    size_t id;
    size_t query;
    size_t k;
    size_t offset;
    size_t count;
};

std::vector<Case> readCases(const std::string &path, std::span<const uint64_t> filters,
                            size_t query_count, size_t documents) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open cases");
    }
    std::vector<Case> cases;
    std::set<size_t> ids;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::vector<std::string> words;
        for (std::string word; fields >> word;) {
            words.push_back(word);
        }
        if (words.size() != 5) {
            throw std::runtime_error("Each case needs five numeric fields");
        }
        Case c{unsignedValue(words[0], true), unsignedValue(words[1], true),
               unsignedValue(words[2]), unsignedValue(words[3], true),
               unsignedValue(words[4], true)};
        if (c.query >= query_count || c.offset > filters.size() ||
            c.count > filters.size() - c.offset || !ids.insert(c.id).second) {
            throw std::runtime_error("Invalid query, filter slice, or duplicate case ID");
        }
        auto eligible = filters.subspan(c.offset, c.count);
        for (size_t i = 0; i < eligible.size(); ++i) {
            if (eligible[i] >= documents || (i && eligible[i] <= eligible[i - 1])) {
                throw std::runtime_error("Eligible IDs must be sorted, unique, and in range");
            }
        }
        cases.push_back(c);
    }
    if (!input.eof() || cases.empty()) {
        throw std::runtime_error("Empty or unreadable cases file");
    }
    return cases;
}

std::vector<CandidateAdapter> selectAdapters(const std::string &policies) {
    const auto registered = candidateAdapters();
    std::vector<CandidateAdapter> selected;
    std::set<std::string> names;
    std::istringstream input(policies);
    for (std::string name; std::getline(input, name, ',');) {
        auto found = std::find_if(registered.begin(), registered.end(),
                                  [&](const auto &adapter) { return adapter.name == name; });
        if (found == registered.end() || !names.insert(name).second) {
            throw std::runtime_error("Unknown or repeated candidate policy: " + name);
        }
        selected.push_back(*found);
    }
    if (selected.empty() || policies.back() == ',') {
        throw std::runtime_error("Empty candidate policy");
    }
    return selected;
}

void optionalCount(std::ostream &output, std::optional<size_t> value) {
    if (value) {
        output << *value;
    } else {
        output << "null";
    }
}

void emit(std::ostream &output, const Case &c, size_t repeat, const std::string &policy,
          const SearchResult &result, double latency_ms, const std::string &error) {
    output << std::setprecision(17) << "{\"case_index\":" << c.id << ",\"repeat\":" << repeat
           << ",\"requested_policy\":" << jsonString(policy)
           << ",\"effective_policy\":" << jsonString(policy)
           << ",\"status\":" << jsonString(result.status) << ",\"ids\":[";
    for (size_t i = 0; i < result.neighbors.size(); ++i) {
        output << (i ? "," : "") << result.neighbors[i].id;
    }
    output << "],\"distances\":[";
    for (size_t i = 0; i < result.neighbors.size(); ++i) {
        output << (i ? "," : "") << result.neighbors[i].distance;
    }
    output << "],\"latency_ms\":" << latency_ms << ",\"counters\":{\"distance_evaluations\":";
    optionalCount(output, result.distance_evaluations);
    output << ",\"candidates_returned\":";
    optionalCount(output, result.candidates_returned);
    output << ",\"batches\":";
    optionalCount(output, result.batches);
    output << '}';
    if (!error.empty()) {
        output << ",\"error\":" << jsonString(error);
    }
    output << "}\n";
}

void run(int argc, char **argv) {
    Arguments args(argc, argv,
                   {"--queries", "--query-count", "--documents", "--dimensions", "--metric",
                    "--graph", "--filters", "--cases", "--output", "--policies", "--repeats",
                    "--ef-runtime", "--batch-size", "--timeout-ms"});
    const size_t documents = args.number("--documents");
    if (documents >= std::numeric_limits<idType>::max()) {
        throw std::runtime_error("Document count exceeds the HNSW internal ID range");
    }
    const size_t dimensions = args.number("--dimensions");
    const size_t query_count = args.number("--query-count");
    const auto metric = parseMetric(args.get("--metric"));
    const size_t repeats = args.number("--repeats");
    const size_t timeout_ms = args.number("--timeout-ms");
    if (timeout_ms > 86400000) {
        throw std::runtime_error("Per-query timeout must not exceed one day");
    }
    Parameters parameters{args.number("--ef-runtime"), args.number("--batch-size"), metric, {}};
    auto adapters = selectAdapters(args.get("--policies"));
    const fs::path output_path(args.get("--output"));
    if (fs::exists(output_path)) {
        throw std::runtime_error("Refusing to overwrite results");
    }
    auto queries = readArray<float>(args.get("--queries"), product(query_count, dimensions));
    validateVectors(queries, dimensions, metric);
    const auto filter_bytes = fs::file_size(args.get("--filters"));
    if (filter_bytes % sizeof(uint64_t)) {
        throw std::runtime_error("Filter payload is not a uint64 array");
    }
    auto filters = readArray<uint64_t>(args.get("--filters"), filter_bytes / sizeof(uint64_t));
    auto cases = readCases(args.get("--cases"), filters, query_count, documents);
    auto begin = Clock::now();
    Index index(HNSWFactory::NewIndex(args.get("--graph")), &VecSimIndex_Free);
    const double load_ms = elapsedMs(begin);
    validateIndex(index.get(), documents, dimensions, metric);
    VecSim_SetTimeoutCallbackFunction(deadlineExpired);
    std::ofstream output(output_path);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    std::vector<uint8_t> membership(documents, 0);
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        for (const auto &c : cases) {
            const auto ids = std::span<const uint64_t>(filters).subspan(c.offset, c.count);
            for (uint64_t id : ids) {
                membership[id] = 1;
            }
            const Eligibility eligible{ids, membership};
            const auto query =
                std::span<const float>(queries).subspan(c.query * dimensions, dimensions);
            for (size_t offset = 0; offset < adapters.size(); ++offset) {
                const auto &adapter = adapters[(repeat + offset) % adapters.size()];
                SearchResult result;
                std::string error;
                begin = Clock::now();
                parameters.deadline = begin + std::chrono::milliseconds(timeout_ms);
                try {
                    result = adapter.search(index.get(), query, eligible, c.k, parameters);
                } catch (const std::exception &exception) {
                    result = {};
                    result.status = "error";
                    error = exception.what();
                }
                const double latency_ms = elapsedMs(begin);
                if (result.status == "ok" && Clock::now() >= parameters.deadline) {
                    result.status = "timeout";
                    result.neighbors.clear();
                }
                emit(output, c, repeat, adapter.name, result, latency_ms, error);
            }
            for (uint64_t id : ids) {
                membership[id] = 0;
            }
        }
    }
    output.close();
    std::cout << std::setprecision(17) << '{';
    provenance(std::cout);
    std::cout << ",\"load_ms\":" << load_ms
              << ",\"index_bytes\":" << VecSimIndex_StatsInfo(index.get()).memory
              << ",\"timing_scope\":\"candidate call including query preparation, traversal, "
                 "filtering, top-k selection and cleanup; excludes graph load, eligibility setup "
                 "and output serialization\",\"filter_setup_scope\":\"one dense byte membership "
                 "array reused across cases; populate and clear outside timed calls\"}\n";
}

void describe() {
    std::cout << '{';
    provenance(std::cout);
    std::cout << ",\"policies\":[";
    bool first = true;
    for (const auto &adapter : candidateAdapters()) {
        std::cout << (first ? "" : ",") << jsonString(adapter.name);
        first = false;
    }
    std::cout << "]}\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        VecSim_SetLogCallbackFunction(logToStderr);
        VecSim_UpdateThreadPoolSize(0);
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error("The native runner currently requires a little-endian host");
        }
        if (argc < 2) {
            throw std::runtime_error("Expected build, run, or describe command");
        }
        const std::string command(argv[1]);
        if (command == "build") {
            build(argc, argv);
        } else if (command == "run") {
            run(argc, argv);
        } else if (command == "describe" && argc == 2) {
            describe();
        } else {
            throw std::runtime_error("Expected build, run, or describe command");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "native_filtered_bench: " << error.what() << '\n';
        return 1;
    }
}
