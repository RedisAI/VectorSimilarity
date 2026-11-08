/* Copyright (c) 2006-Present, Redis Ltd. All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3. */
// Bounded validation driver. Exhaustive compressed scans below are codec diagnostics,
// not a supported FLAT index or a measurement of public product query latency.
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/quantization/turboquant_components.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace selected_modes {
using Graph = HNSWIndex<float, float>;
using Tiered = TieredHNSWIndex<float, float>;
using Owner = std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)>;
using Ranking = std::vector<std::pair<double, size_t>>;
using Blob = MemoryUtils::unique_blob;
using namespace vecsim::quantization;
constexpr size_t dim = 768, k = 10, query_offset = 500000;
constexpr uint64_t seed = 42;
volatile double sink = 0;
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
void escape(const void *p) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(p) : "memory");
#else
    sink = static_cast<const unsigned char *>(p)[0];
#endif
}
uint64_t ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
uint64_t hash(const void *ptr, size_t bytes, uint64_t h = UINT64_C(14695981039346656037)) {
    const auto *p = static_cast<const unsigned char *>(ptr);
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
size_t number(const std::string &s) {
    require(!s.empty() && s.find_first_not_of("0123456789") == std::string::npos,
            "Expected unsigned decimal integer: " + s);
    size_t pos = 0;
    const auto n = std::stoull(s, &pos);
    require(pos == s.size() && n <= std::numeric_limits<size_t>::max(), "Integer overflow");
    return static_cast<size_t>(n);
}
// Standalone process only: the normal VecSim hooks are libc. Bookkeeping uses
// std::unordered_map's ordinary allocator, so it never enters these hooks.
class TransientMemoryProbe {
    inline static TransientMemoryProbe *active_ = nullptr;
    std::unordered_map<std::uintptr_t, size_t> allocations_;
    size_t live_ = 0, peak_ = 0;
    bool failed_ = false, installed_ = false;

    void remember(void *pointer, size_t bytes) noexcept {
        if (!pointer) return;
        try {
            const bool inserted = allocations_.emplace(reinterpret_cast<std::uintptr_t>(pointer), bytes).second;
            if (!inserted || bytes > std::numeric_limits<size_t>::max() - live_) {
                failed_ = true;
                return;
            }
            live_ += bytes;
            peak_ = std::max(peak_, live_);
        } catch (...) {
            // Do not throw through allocation callbacks or replace successful
            // user allocations. Reject incomplete accounting after restoring hooks.
            failed_ = true;
        }
    }
    void forget(std::uintptr_t address) noexcept {
        const auto found = allocations_.find(address);
        if (found == allocations_.end()) return; // Allocated before this probe.
        live_ -= found->second;
        allocations_.erase(found);
    }
    static void *allocate(size_t bytes) {
        void *pointer = std::malloc(bytes);
        active_->remember(pointer, bytes);
        return pointer;
    }
    static void *callocate(size_t count, size_t width) {
        void *pointer = std::calloc(count, width);
        if (pointer && width != 0 && count > std::numeric_limits<size_t>::max() / width)
            active_->failed_ = true;
        else active_->remember(pointer, count * width);
        return pointer;
    }
    static void *reallocate(void *pointer, size_t bytes) {
        const auto old_address = reinterpret_cast<std::uintptr_t>(pointer);
        if (bytes == 0) {
            // Explicitly use libc's permitted free-and-null zero-size behavior.
            active_->forget(old_address);
            std::free(pointer);
            return nullptr;
        }
        void *replacement = std::realloc(pointer, bytes);
        if (replacement) {
            active_->forget(old_address);
            active_->remember(replacement, bytes);
        }
        return replacement;
    }
    static void release(void *pointer) {
        active_->forget(reinterpret_cast<std::uintptr_t>(pointer));
        std::free(pointer);
    }
public:
    TransientMemoryProbe() {
        require(active_ == nullptr, "Nested transient memory probe");
        active_ = this;
        VecSim_SetMemoryFunctions({allocate, callocate, reallocate, release});
        installed_ = true;
    }
    ~TransientMemoryProbe() { restore(); }
    void restore() noexcept {
        if (!installed_) return;
        VecSim_SetMemoryFunctions({std::malloc, std::calloc, std::realloc, std::free});
        installed_ = false;
        active_ = nullptr;
    }
    size_t peak() const { return peak_; }
    size_t live() const { return live_; }
    size_t liveAllocations() const { return allocations_.size(); }
    bool complete() const { return !failed_; }
};

struct Args {
    std::string stage, codec, training = "full", fixture;
    std::filesystem::path output;
    size_t vectors = 0, queries = 0;
};
Args parse(int argc, char **argv) {
    Args a;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        require(seen.insert(key).second && i + 1 < argc, "Duplicate/missing option: " + key);
        const std::string value = argv[++i];
        if (key == "--stage") a.stage = value;
        else if (key == "--codec") a.codec = value;
        else if (key == "--training") a.training = value;
        else if (key == "--fixture" || key == "--source") a.fixture = value;
        else if (key == "--output") a.output = value;
        else if (key == "--vectors") a.vectors = number(value);
        else if (key == "--queries") a.queries = number(value);
        else throw std::invalid_argument("Unknown option: " + key);
    }
    require(a.stage == "micro" || a.stage == "tiny" || a.stage == "representation" ||
            a.stage == "hnsw", "--stage must be micro|tiny|representation|hnsw");
    require(a.codec == "tq2" || a.codec == "tq3",
            "--codec must be tq2|tq3");
    require(!a.output.empty(), "--output NEW_DIRECTORY is required");
    if (a.stage == "micro") {
        require(a.vectors == 0 && a.queries == 0, "Micro sizes are fixed: omit --vectors/--queries");
        a.vectors = 512; a.queries = 16;
    } else {
        require(!a.fixture.empty(), "--fixture HNSW_V3_FILE is required");
        if (!a.vectors) a.vectors = a.stage == "tiny" ? 1000 : 100000;
        if (!a.queries) a.queries = a.vectors == 1000 ? 5 : a.vectors == 10000 ? 100 : 500;
        require((a.stage == "tiny" && ((a.vectors == 1000 && a.queries == 5) ||
                                      (a.vectors == 10000 && a.queries == 100))) ||
                (a.stage != "tiny" && a.vectors == 100000 && a.queries == 500),
                "Bounded cases: tiny1000/5, tiny10000/100, representation/hnsw100000/500");
    }
    if (a.training != "full") {
        const size_t n = number(a.training);
        require(n > 0 && n <= 100000, "Training count must be 1..100000 or full");
    }
    require(!std::filesystem::exists(a.output), "Output directory must be new");
    return a;
}
struct Output {
    std::ofstream config, checks, memory, timings, truth, results, accuracy, numerical;
    explicit Output(const Args &a) {
        require(std::filesystem::create_directories(a.output), "Cannot create output directory");
        auto open = [&](std::ofstream &f, const char *name) {
            f.exceptions(std::ios::failbit | std::ios::badbit);
            f.open(a.output / name); f << std::setprecision(17);
        };
        open(config,"config.tsv"); open(checks,"checks.tsv"); open(memory,"memory.tsv");
        open(timings,"timings.tsv"); open(truth,"truth.tsv"); open(results,"results.tsv");
        open(accuracy,"accuracy.tsv"); open(numerical,"numerical.tsv");
        config << "key\tvalue\n";
        checks << "check\tstatus\tdetail\n";
        memory << "scope\tfield\tbytes_or_count\n";
        timings << "scope\toperation\tef\tbatch\toperations\telapsed_ns\tmean_ns\n";
        truth << "query\tsource_id\trank\tlabel\tscore\n";
        results << "scope\tef\tquery\tsource_id\trank\tlabel\tscore\n";
        accuracy << "scope\tef\tquery\thits_at_10\trecall_at_10\n";
        numerical << "scope\trow\tquery\tactual\tscalar_reference\tabsolute_error\n";
        config << "stage\t" << a.stage << "\ncodec\t" << a.codec << "\ntraining_requested\t"
               << a.training << "\nvectors\t" << a.vectors << "\nqueries\t" << a.queries
               << "\ndim\t768\nmetric\tIP_on_public_normalized_cosine_rows\nk\t10\nM\t16"
               << "\nefConstruction\t100\nef_values\t40,80,160\ncodec_seed\t42"
               << "\ngraph_seed\t100\nblock_size\t1024\nquery_offset\t500000"
               << "\ncompiler\t" << __VERSION__
               << "\nfixture_expected_sha256\tb59f5311786cb48d7f697d11722a4042d953eb174254f0655851737d639cd379"
               << "\nfixture_sha256_validation\texternal_runner_required"
               << "\nfixture_load_scope\texcluded_from_selected_build_query_and_index_memory"
               << "\ntruth_scope\tindependent_double_scalar_FP32_inner_product"
               << "\nrepresentation_scope\tcodec_diagnostic_not_supported_product_FLAT_not_product_timing"
               << "\nproduct_timing_scope\tpublic_TopK_preparation_search_reply_and_iterator_cleanup"
               << "\ntiming_statistics\tthree_within_process_batch_means_no_percentiles"
               << "\nindex_memory_scope\tVecSim_requested_allocations"
               << "\nmemory_excluded\tharness_base_queries_training_mean_truth_rankings_and_libc_bookkeeping\n";
    }
    void check(const std::string &name, bool ok, const std::string &detail = "") {
        checks << name << '\t' << (ok ? "PASS" : "FAIL") << '\t' << detail << '\n'; checks.flush();
        require(ok, "Check failed: " + name + " " + detail);
    }
    void time(const std::string &scope, const std::string &operation, size_t ef, size_t batch,
              size_t count, uint64_t elapsed) {
        timings << scope << '\t' << operation << '\t' << ef << '\t' << batch << '\t'
                << count << '\t' << elapsed << '\t' << double(elapsed) / count << '\n';
    }
};
struct Data {
    std::vector<float> base, queries;
    explicit Data(const Args &a, Output &out) : base(a.vectors * dim), queries(a.queries * dim) {
        uint64_t raw_base = UINT64_C(14695981039346656037), raw_queries = raw_base;
        if (a.stage == "micro") {
            auto fill = [](float *row, size_t id) {
                for (size_t d = 0; d < dim; ++d)
                    row[d] = float(std::sin((d + 1) * (id + 1) * .713) +
                                   .3 * std::cos((d + 3) * (id + 2) * .419) + .15);
                VecSim_Normalize(row, dim, VecSimType_FLOAT32);
            };
            for (size_t i = 0; i < a.vectors; ++i) fill(base.data() + i * dim, i);
            for (size_t q = 0; q < a.queries; ++q) fill(queries.data() + q * dim, query_offset + q);
            raw_base = hash(base.data(),base.size() * sizeof(float));
            raw_queries = hash(queries.data(),queries.size() * sizeof(float));
        } else {
            // Keep the public reader as the source of truth for the historical v3 layout.
            // The source owner dies before any truth, codec, or selected graph is allocated.
            Owner owner(HNSWFactory::NewIndex(a.fixture), VecSimIndex_Free);
            auto *source = dynamic_cast<Graph *>(owner.get());
            require(source, "Expected FP32 HNSW fixture");
            const auto info = source->basicInfo();
            out.check("fixture_metadata", info.type == VecSimType_FLOAT32 && info.dim == dim &&
                info.metric == VecSimMetric_Cosine && !info.isMulti &&
                source->quantType == VecSimQuant_NONE && source->indexSize() >= query_offset + a.queries &&
                source->getStoredDataSize() == dim * sizeof(float) &&
                source->getVersion() == HNSWSerializer::EncodingVersion::V3);
            auto copy = [&](size_t id, float *row, uint64_t &raw_hash) {
                require(!source->isMarkedDeleted(id) && !source->isInProcess(id), "Deleted/in-process fixture row");
                require(source->getExternalLabel(id) == id,"Fixture row identity is not its corpus label");
                std::memcpy(row, source->getDataByInternalId(id), dim * sizeof(float));
                raw_hash = hash(row, dim * sizeof(float), raw_hash);
                double norm = 0;
                for (size_t d = 0; d < dim; ++d) {
                    require(std::isfinite(row[d]), "Nonfinite fixture coordinate");
                    norm += double(row[d]) * row[d];
                }
                require(std::abs(norm - 1) <= 1e-3, "Fixture cosine row is not normalized");
                VecSim_Normalize(row, dim, VecSimType_FLOAT32);
            };
            for (size_t i = 0; i < a.vectors; ++i) copy(i, base.data() + i * dim, raw_base);
            for (size_t q = 0; q < a.queries; ++q) copy(query_offset + q, queries.data() + q * dim, raw_queries);
            out.config << "fixture\t" << std::filesystem::absolute(a.fixture).string()
                       << "\nfixture_vectors\t" << source->indexSize() << '\n';
        }
        // Hash buckets avoid retaining another 100000*768*4-byte copy solely for uniqueness.
        std::unordered_multimap<uint64_t,size_t> base_hashes;
        for (size_t i = 0; i < a.vectors; ++i)
            base_hashes.emplace(hash(row(i), dim * 4), i);
        std::set<std::string> unique_queries;
        for (size_t q = 0; q < a.queries; ++q) {
            const auto *v = query(q);
            require(unique_queries.emplace(reinterpret_cast<const char *>(v), dim * 4).second,
                    "Duplicate query contents");
            const auto [first, last] = base_hashes.equal_range(hash(v, dim * 4));
            for (auto it = first; it != last; ++it)
                require(std::memcmp(v, row(it->second), dim * 4) != 0, "Query contents duplicate base");
            out.config << "query_" << q << "_normalized_fnv1a64\t" << hash(v, dim * 4) << '\n';
        }
        out.config << "raw_base_fnv1a64\t" << raw_base << "\nraw_queries_fnv1a64\t" << raw_queries
                   << "\nnormalized_base_fnv1a64\t" << hash(base.data(), base.size() * 4)
                   << "\nnormalized_queries_fnv1a64\t" << hash(queries.data(), queries.size() * 4)
                   << "\nsource_released_before_selected_allocations\t1\n";
        out.check("heldout_content_disjoint", true);
    }
    const float *row(size_t i) const { return base.data() + i * dim; }
    const float *query(size_t q) const { return queries.data() + q * dim; }
    std::vector<float> mean(size_t count) const {
        std::vector<double> sum(dim);
        for (size_t i = 0; i < count; ++i)
            for (size_t d = 0; d < dim; ++d) sum[d] += row(i)[d];
        std::vector<float> result(dim);
        for (size_t d = 0; d < dim; ++d) result[d] = float(sum[d] / count);
        return result;
    }
};
float readFloat(const void *record, size_t offset) {
    float f; std::memcpy(&f, static_cast<const unsigned char *>(record) + offset, 4); return f;
}
struct Codec {
    std::string name;
    std::shared_ptr<VecSimAllocator> allocator = VecSimAllocator::newVecsimAllocator();
    std::shared_ptr<TurboQuant> tq;
    std::unique_ptr<IndexCalculatorInterface<float>> calculator;
    std::unique_ptr<PreprocessorsContainerAbstract> preprocessors;
    std::vector<float> mean;
    size_t stored_bytes = 0, query_bytes = 0;
    explicit Codec(const std::string &name_, const std::vector<float> &mean_) : name(name_), mean(mean_) {
        require(name == "tq2" || name == "tq3", "Unknown codec: " + name);
        tq = std::allocate_shared<TurboQuant>(VecsimSTLAllocator<TurboQuant>(allocator), allocator,
            dim, name == "tq2" ? 2 : 3, seed, TurboQuantVariant::MSE,
            TurboQuantExecution::Dispatch, TurboQuantScalePolicy::ProjectionMatch,
            TurboQuantBiasPolicy::MeanError, std::span<const float>(mean));
        stored_bytes = tq->storageBytes(); query_bytes = tq->queryBytes() + 4;
        auto parts = createTurboQuantComponents(allocator, tq);
        calculator.reset(parts.indexCalculator); preprocessors.reset(parts.preprocessors);
    }
    Blob encode(const float *x) { return preprocessors->preprocessForStorage(x, dim * 4); }
    Blob prepare(const float *q) { return preprocessors->preprocessQuery(q, dim * 4); }
    float score(const void *x, const void *q) const {
        return calculator->calcDistanceForQuery(x,q,dim);
    }
    float pair(const void *x, const void *y) const { return calculator->calcDistance(x,y,dim); }
    // Independent unpacking and FP64 arithmetic, without calling the scorer or decoder.
    std::vector<double> decoded(const void *record) const {
        const auto *b = static_cast<const unsigned char *>(record);
        std::vector<double> result(dim);
        const unsigned bits = tq->bits();
        const double scale = readFloat(record, (dim * bits + 7) / 8);
        for (size_t d = 0; d < dim; ++d) {
            const size_t bit = d * bits;
            unsigned code = 0;
            for (unsigned j = 0; j < bits; ++j)
                code |= unsigned((b[(bit + j) / 8] >> ((bit + j) % 8)) & 1) << j;
            result[d] = scale * tq->centroids()[code];
        }
        return result;
    }
    double scalarScore(const void *record, const void *prepared, const float *raw_query) const {
        const auto x = decoded(record);
        double dot = 0;
        for (size_t d = 0; d < dim; ++d) dot += x[d] * readFloat(prepared, d * 4) + double(mean[d]) * raw_query[d];
        return 1 - dot + readFloat(record, stored_bytes - 4);
    }
    double scalarPair(const void *a, const void *b) const {
        const auto x = decoded(a), y = decoded(b);
        double dot = 0;
        for (size_t d = 0; d < dim; ++d) {
            const double m = tq->biasRotatedMean()[d];
            dot += (x[d] + m) * (y[d] + m);
        }
        const double correction = double(readFloat(a, stored_bytes - 4)) + readFloat(b, stored_bytes - 4);
        return 1 - dot + correction;
    }
    void memory(Output &out, size_t records) {
        const size_t before = allocator->getAllocationSize();
        std::vector<float> q(dim, .01f);
        auto blob = prepare(q.data());
        out.memory << "codec\trecord_bytes\t" << stored_bytes << "\ncodec\trecord_count\t" << records
                   << "\ncodec\trecord_payload_total\t" << records * stored_bytes
                   << "\ncodec\tshared_allocator_requested\t" << before
                   << "\ncodec\tprepared_query_payload\t" << query_bytes
                   << "\ncodec\tprepared_query_allocator_delta\t" << allocator->getAllocationSize() - before
                   << "\ncodec\tshared_float_payload\t" << tq->sharedDataBytes()
                   << "\ncodec\tFP32_retained_records\t0\n";
        out.memory << "codec\tencoding_preprocessor_residual_payload\t" << dim * sizeof(float)
                   << "\ncodec\tencoding_codec_unit_scratch_payload\t" << dim * sizeof(float)
                   << "\ncodec\tencoding_known_temporary_payload_total\t" << 2 * dim * sizeof(float) << '\n';
        out.config << "TQ_encoding_temporary_scope\tpreprocessor_residual_plus_MSE_codec_unit_vectors_excludes_output_record_allocator_headers_and_stack_scalars\n";
        out.config << "query_kernel\t" << tq->scoringKernelName()
                   << "\ngraph_kernel\t" << tq->storedScoringKernelName() << '\n';
    }
};
Ranking topK(Ranking values) {
    require(values.size() >= k, "Insufficient candidates");
    std::partial_sort(values.begin(), values.begin() + k, values.end()); values.resize(k); return values;
}
std::vector<Ranking> truth(const Args &a, const Data &data, Output &out) {
    std::vector<Ranking> result;
    for (size_t q = 0; q < a.queries; ++q) {
        Ranking all; all.reserve(a.vectors);
        for (size_t i = 0; i < a.vectors; ++i) {
            double dot = 0;
            for (size_t d = 0; d < dim; ++d) dot += double(data.row(i)[d]) * data.query(q)[d];
            all.emplace_back(1 - dot, i);
        }
        result.push_back(topK(std::move(all)));
        for (size_t rank = 0; rank < k; ++rank)
            out.truth << q << '\t' << query_offset + q << '\t' << rank + 1 << '\t'
                      << result.back()[rank].second << '\t' << result.back()[rank].first << '\n';
    }
    out.truth.flush(); return result;
}
void writeRanking(Output &out, const std::string &scope, size_t ef, size_t q,
                  const Ranking &ranking, const Ranking &expected) {
    size_t hits = 0;
    for (size_t rank = 0; rank < ranking.size(); ++rank) {
        for (const auto &match : expected) hits += match.second == ranking[rank].second;
        out.results << scope << '\t' << ef << '\t' << q << '\t' << query_offset + q << '\t'
                    << rank + 1 << '\t' << ranking[rank].second << '\t' << ranking[rank].first << '\n';
    }
    out.accuracy << scope << '\t' << ef << '\t' << q << '\t' << hits << '\t' << double(hits) / k << '\n';
}
void numerical(Codec &codec, const Data &data, Output &out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        auto x = codec.encode(data.row(i)), y = codec.encode(data.row((i + 1) % count));
        const size_t q = i % (data.queries.size() / dim);
        auto prepared = codec.prepare(data.query(q));
        const double actual = codec.score(x.get(), prepared.get());
        const double expected = codec.scalarScore(x.get(), prepared.get(), data.query(q));
        const double pair_actual = codec.pair(x.get(), y.get()), pair_expected = codec.scalarPair(x.get(),y.get());
        out.numerical << "query\t" << i << '\t' << q << '\t' << actual << '\t' << expected << '\t' << std::abs(actual - expected)
                      << "\npair\t" << i << '\t' << q << '\t' << pair_actual << '\t' << pair_expected << '\t' << std::abs(pair_actual - pair_expected) << '\n';
        require(std::isfinite(actual) && std::abs(actual - expected) <= 3e-5 * (1 + std::abs(expected)),
                "Independent scalar query reference mismatch");
        require(std::isfinite(pair_actual) && std::abs(pair_actual - pair_expected) <= 3e-5 * (1 + std::abs(pair_expected)),
                "Independent scalar graph reference mismatch");
        require(std::abs(pair_actual - codec.pair(y.get(),x.get())) < 3e-5, "Asymmetric graph scorer");
    }
    out.check("independent_scalar_query_and_graph", true, std::to_string(count) + " records");
}
// Odd dimensions exercise packed-code tails independently of the SIMD scorers.
void tailReferences(const std::string &name, Output &out) {
    const std::vector<size_t> dimensions{3,31,33,65};
    for (size_t d : dimensions) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        std::vector<float> x(d), y(d), q(d), mean(d);
        for (size_t i = 0; i < d; ++i) {
            x[i] = float(std::sin((i + 1) * .713)); y[i] = float(std::cos((i + 3) * .419));
            q[i] = float(std::sin((i + 7) * .831)); mean[i] = .1f * x[i];
        }
        VecSim_Normalize(x.data(),d,VecSimType_FLOAT32);
        VecSim_Normalize(y.data(),d,VecSimType_FLOAT32);
        VecSim_Normalize(q.data(),d,VecSimType_FLOAT32);
        std::shared_ptr<TurboQuant> tq;
        std::unique_ptr<IndexCalculatorInterface<float>> calculator;
        std::unique_ptr<PreprocessorsContainerAbstract> prep;
        size_t stored = 0;
        require(name == "tq2" || name == "tq3", "Unknown codec: " + name);
        tq = std::make_shared<TurboQuant>(allocator,d,name == "tq2" ? 2 : 3,seed,
            TurboQuantVariant::MSE,TurboQuantExecution::Dispatch,
            TurboQuantScalePolicy::ProjectionMatch,TurboQuantBiasPolicy::MeanError,mean);
        auto parts = createTurboQuantComponents(allocator,tq);
        calculator.reset(parts.indexCalculator); prep.reset(parts.preprocessors); stored = tq->storageBytes();
        auto a = prep->preprocessForStorage(x.data(),d * 4), b = prep->preprocessForStorage(y.data(),d * 4);
        auto query = prep->preprocessQuery(q.data(),d * 4);
        auto decode = [&](const void *record) {
            std::vector<double> result(d);
            const auto *bytes = static_cast<const unsigned char *>(record);
            for (size_t i = 0; i < d; ++i) {
                unsigned code = 0;
                for (unsigned bit = 0; bit < tq->bits(); ++bit) {
                    const size_t offset = i * tq->bits() + bit;
                    code |= unsigned((bytes[offset / 8] >> (offset % 8)) & 1) << bit;
                }
                result[i] = double(readFloat(record,stored - 8)) * tq->centroids()[code];
            }
            return result;
        };
        const auto da = decode(a.get()), db = decode(b.get());
        double ip = 0, pair_ip = 0, mean_ip = 0;
        for (size_t i = 0; i < d; ++i) {
            const double query_value = readFloat(query.get(),i * 4);
            ip += da[i] * query_value;
            const double m = tq->biasRotatedMean()[i];
            pair_ip += (da[i] + m) * (db[i] + m);
            mean_ip += double(mean[i]) * q[i];
        }
        double expected = 1 - ip, pair_expected = 1 - pair_ip;
        expected += readFloat(a.get(),stored - 4) - mean_ip;
        pair_expected += double(readFloat(a.get(),stored - 4)) + readFloat(b.get(),stored - 4);
        const double actual = calculator->calcDistanceForQuery(a.get(),query.get(),d);
        const double pair_actual = calculator->calcDistance(a.get(),b.get(),d);
        out.numerical << "tail_query_dim" << d << "\t0\t0\t" << actual << '\t' << expected << '\t' << std::abs(actual - expected)
                      << "\ntail_pair_dim" << d << "\t0\t0\t" << pair_actual << '\t' << pair_expected << '\t' << std::abs(pair_actual - pair_expected) << '\n';
        out.check("tail_scalar_reference_dim" + std::to_string(d),
            std::abs(actual - expected) <= 3e-5 * (1 + std::abs(expected)) &&
            std::abs(pair_actual - pair_expected) <= 3e-5 * (1 + std::abs(pair_expected)));
    }
}
void micro(const Args &a, const Data &data, Codec &codec, Output &out) {
    tailReferences(a.codec,out);
    numerical(codec,data,out,64);
    codec.memory(out,a.vectors);
    std::vector<Blob> records, queries;
    for (size_t i = 0; i < a.vectors; ++i) records.push_back(codec.encode(data.row(i)));
    for (size_t q = 0; q < a.queries; ++q) queries.push_back(codec.prepare(data.query(q)));
    auto fp_allocator = VecSimAllocator::newVecsimAllocator();
    auto fp_parts = CreateIndexComponents<float,float>(fp_allocator,VecSimMetric_IP,dim,true);
    std::unique_ptr<IndexCalculatorInterface<float>> fp_calculator(fp_parts.indexCalculator);
    std::unique_ptr<PreprocessorsContainerAbstract> fp_preprocessors(fp_parts.preprocessors);
    const size_t fp_before = fp_allocator->getAllocationSize();
    auto fp_query = fp_preprocessors->preprocessQuery(data.query(0),dim * 4);
    auto fp_record = fp_preprocessors->preprocessForStorage(data.row(0),dim * 4);
    out.memory << "FP32_components\tprepared_query_plus_record_allocator_delta\t"
               << fp_allocator->getAllocationSize() - fp_before << '\n';
    std::mt19937 random(seed);
    std::vector<size_t> order(a.vectors); std::iota(order.begin(),order.end(),0);
    std::shuffle(order.begin(),order.end(),random);
    constexpr size_t iterations = 32768;
    for (size_t batch = 0; batch < 3; ++batch) {
        auto measure = [&](const std::string &op, size_t n, auto fn) {
            double total = 0; const auto start = ns();
            for (size_t i = 0; i < n; ++i) total += fn(i);
            const auto elapsed = ns() - start; sink = total;
            out.time("full_record_micro",op,0,batch,n,elapsed);
        };
        measure("query_hot",iterations,[&](size_t) { return codec.score(records[0].get(),queries[0].get()); });
        measure("graph_hot",iterations,[&](size_t) { return codec.pair(records[0].get(),records[1].get()); });
        measure("query_randomized",iterations,[&](size_t i) {
            return codec.score(records[order[i % a.vectors]].get(),queries[i % a.queries].get()); });
        measure("graph_randomized",iterations,[&](size_t i) {
            return codec.pair(records[order[i % a.vectors]].get(),records[order[(i + 17) % a.vectors]].get()); });
        size_t dependent = 0;
        measure("query_dependent",iterations,[&](size_t i) {
            const float v = codec.score(records[dependent].get(),queries[i % a.queries].get());
            uint32_t bits; std::memcpy(&bits,&v,4); dependent = (bits + i) % a.vectors; return v; });
        measure("graph_dependent",iterations,[&](size_t i) {
            const float v = codec.pair(records[dependent].get(),records[order[i % a.vectors]].get());
            uint32_t bits; std::memcpy(&bits,&v,4); dependent = (bits + i) % a.vectors; return v; });
        measure("encode_with_temporary_and_cleanup",64,[&](size_t i) {
            auto b = codec.encode(data.row(i)); escape(b.get());
            return double(static_cast<const unsigned char *>(b.get())[0]); });
        measure("prepare_query_with_allocation_and_cleanup",64,[&](size_t i) {
            auto q = codec.prepare(data.query(i % a.queries)); escape(q.get());
            return double(static_cast<const unsigned char *>(q.get())[0]); });
        measure("FP32_scalar_query",iterations,[&](size_t i) {
            double dot = 0;
            for (size_t d = 0; d < dim; ++d) dot += double(data.row(order[i % a.vectors])[d]) * data.query(i % a.queries)[d];
            return 1 - dot; });
        measure("FP32_native_query_hot",iterations,[&](size_t) {
            return fp_calculator->calcDistanceForQuery(fp_record.get(),fp_query.get(),dim); });
        measure("FP32_component_storage_prepare_and_cleanup",64,[&](size_t i) {
            auto b = fp_preprocessors->preprocessForStorage(data.row(i),dim * 4);
            escape(b.get()); return double(static_cast<const float *>(b.get())[0]); });
        measure("FP32_component_query_prepare_and_cleanup",64,[&](size_t i) {
            auto q = fp_preprocessors->preprocessQuery(data.query(i % a.queries),dim * 4);
            escape(q.get()); return double(static_cast<const float *>(q.get())[0]); });
    }
    out.check("micro_complete",true);
}
void representation(const Args &a, const Data &data, Codec &codec, Output &out,
                    const std::vector<Ranking> &expected) {
    numerical(codec,data,out,64); codec.memory(out,a.vectors);
    // A single contiguous diagnostic record table has no retained FP32 records.
    std::vector<unsigned char> records(a.vectors * codec.stored_bytes);
    const auto start = ns();
    for (size_t i = 0; i < a.vectors; ++i) {
        auto record = codec.encode(data.row(i));
        std::memcpy(records.data() + i * codec.stored_bytes,record.get(),codec.stored_bytes);
    }
    out.time("codec_diagnostic_not_product","encode_all",0,0,a.vectors,ns() - start);
    out.config << "diagnostic_record_fnv1a64\t" << hash(records.data(),records.size()) << '\n';
    for (size_t q = 0; q < a.queries; ++q) {
        // Exactly one prepared query per exhaustive scan.
        auto prepared = codec.prepare(data.query(q));
        Ranking all; all.reserve(a.vectors);
        for (size_t i = 0; i < a.vectors; ++i)
            all.emplace_back(codec.score(records.data() + i * codec.stored_bytes,prepared.get()),i);
        writeRanking(out,"codec_exhaustive_diagnostic",0,q,topK(std::move(all)),expected[q]);
    }
    out.check("representation_complete",true,"not_product_FLAT");
}
const char *recordByLabel(Graph *graph, size_t label) {
    const auto ids = graph->getElementIds(label);
    require(ids.size() == 1,"Expected one internal row for external label");
    return graph->getDataByInternalId(ids.front());
}
HNSWParams params(const std::string &codec, const std::vector<float> &mean,
                  VecSimTurboQuantParams &config) {
    config = {.mean = mean.data(), .seed = seed};
    return {.type = VecSimType_FLOAT32, .dim = dim, .metric = VecSimMetric_IP,
            .multi = false, .blockSize = 1024, .M = 16, .efConstruction = 100, .efRuntime = 80,
            .quantType = codec == "tq2" ? VecSimQuant_TQ2 : VecSimQuant_TQ3,
            .quantParams = &config};
}
Ranking query(VecSimIndex *index, const float *q, size_t ef) {
    VecSimQueryParams p{}; p.hnswRuntimeParams.efRuntime = ef;
    auto *reply = VecSimIndex_TopKQuery(index,q,k,&p,BY_SCORE);
    require(reply && VecSimQueryReply_GetCode(reply) == VecSim_QueryReply_OK, "TopK failed");
    auto *it = VecSimQueryReply_GetIterator(reply);
    Ranking result;
    while (VecSimQueryReply_IteratorHasNext(it)) {
        auto *item = VecSimQueryReply_IteratorNext(it);
        const double score = VecSimQueryResult_GetScore(item);
        require(std::isfinite(score),"Nonfinite result");
        result.emplace_back(score,VecSimQueryResult_GetId(item));
    }
    VecSimQueryReply_IteratorFree(it); VecSimQueryReply_Free(reply);
    require(result.size() == k,"Wrong result count"); return result;
}
// Timing intentionally consumes every result and frees both iterator and reply.
void timedQuery(VecSimIndex *index, const float *q, size_t ef) {
    VecSimQueryParams p{}; p.hnswRuntimeParams.efRuntime = ef;
    auto *reply = VecSimIndex_TopKQuery(index,q,k,&p,BY_SCORE);
    require(reply && VecSimQueryReply_GetCode(reply) == VecSim_QueryReply_OK, "Timed TopK failed");
    auto *it = VecSimQueryReply_GetIterator(reply);
    double total = 0; size_t count = 0;
    while (VecSimQueryReply_IteratorHasNext(it)) {
        const auto *item = VecSimQueryReply_IteratorNext(it);
        total += VecSimQueryResult_GetScore(item); ++count;
    }
    VecSimQueryReply_IteratorFree(it); VecSimQueryReply_Free(reply);
    require(count == k,"Timed result count mismatch"); sink = total;
}
int unexpectedSubmit(void *,void *,AsyncJob **,JobCallback *,size_t) {
    throw std::runtime_error("In-place benchmark unexpectedly submitted asynchronous jobs");
}
void publicGraph(const Args &a, const Data &data, Codec &codec, Output &out,
                 const std::vector<Ranking> &expected, size_t training_count) {
    VecSim_SetWriteMode(VecSim_WriteInPlace);
    Owner owner(nullptr,VecSimIndex_Free);
    Graph *graph = nullptr;
    const std::string scope = "public_tiered_TQ_HNSW";
    const auto start = ns();
    VecSimTurboQuantParams config{};
    HNSWParams hp = params(a.codec,codec.mean,config);
    VecSimParams primary{}; primary.algo = VecSimAlgo_HNSWLIB; primary.algoParams.hnswParams = hp;
    // All TQ builds exercise the supported tiered training path, including a full-base threshold.
    // A seed is explicit, but a supplied diagnostic mean must not initialize the tiered fit.
    config.mean = nullptr;
    TieredIndexParams tp{};
    tp.primaryIndexParams = &primary; tp.flatBufferLimit = 1024; tp.submitCb = unexpectedSubmit;
    tp.specificParams.tieredHnswParams.QuantNormalizationSetSize = training_count;
    tp.specificParams.tieredHnswParams.swapJobThreshold = 1024;
    VecSimParams p{}; p.algo = VecSimAlgo_TIERED; p.algoParams.tieredParams = tp;
    owner.reset(VecSimIndex_New(&p));
    auto *tiered = dynamic_cast<Tiered *>(owner.get());
    require(tiered,"Public tiered factory rejected TQ configuration");
    graph = tiered->getBackendForTests();
    require(owner && graph,"Cannot construct selected index");
    uint64_t observed_allocator_max = owner->getAllocationSize();
    uint64_t pre_threshold_allocator = observed_allocator_max;
    for (size_t i = 0; i < a.vectors; ++i) {
        require(VecSimIndex_AddVector(owner.get(),data.row(i),i) == 1,"Insertion failed");
        const uint64_t observed = owner->getAllocationSize();
        observed_allocator_max = std::max(observed_allocator_max,observed);
        if (i + 1 < training_count) pre_threshold_allocator = observed;
    }
    out.time(scope,"construct_insert_with_after_add_memory_sampling",0,0,a.vectors,ns() - start);
    out.memory << scope << "\tbuild_allocator_observed_after_add_max_lower_bound\t" << observed_allocator_max << '\n';
    out.memory << scope << "\ttraining_pre_threshold_allocator_observed\t" << pre_threshold_allocator
               << '\n' << scope << "\ttraining_threshold_FP32_row_payload\t" << training_count * dim * sizeof(float)
               << '\n' << scope << "\ttraining_running_sum_double_payload\t" << dim * sizeof(double)
               << '\n' << scope << "\ttraining_mean_float_payload\t" << dim * sizeof(float) << '\n';
    out.config << "build_peak_scope\tobserved_after_each_add_lower_bound_excludes_within_call_temporaries_not_exact_peak\n"
               << "training_migration_overlap\tthreshold_FP32_payload_can_overlap_growing_compressed_graph_during_synchronous_migration_not_joint_peak_measured\n";
    out.check("public_size_and_labels",owner->indexSize() == a.vectors && owner->indexLabelCount() == a.vectors && graph->indexSize() == a.vectors);
    out.check("graph_integrity",graph->checkIntegrity().valid_state && graph->getM() == 16 && graph->getEfConstruction() == 100);
    const auto actual = graph->getTurboQuant()->biasMean();
    out.check("tiered_mean_exact",actual.size() == codec.mean.size() &&
        std::memcmp(actual.data(),codec.mean.data(),dim * 4) == 0);
    const auto actual_codec = graph->getTurboQuant();
    const auto same = [](std::span<const float> a, std::span<const float> b) {
        return a.size() == b.size() && std::memcmp(a.data(),b.data(),a.size_bytes()) == 0;
    };
    out.check("public_diagnostic_fitted_state_exact",
        actual_codec->bits() == codec.tq->bits() && actual_codec->seed() == codec.tq->seed() &&
        same(actual_codec->rotation(),codec.tq->rotation()) &&
        same(actual_codec->centroids(),codec.tq->centroids()) &&
        same(actual_codec->biasRotatedMean(),codec.tq->biasRotatedMean()));
    const auto info = graph->debugInfo().hnswInfo;
    out.check("TQ_metadata",info.quantizedVectorBytes == codec.stored_bytes &&
              info.quantizerSharedBytes == graph->getTurboQuant()->sharedDataBytes());
    const size_t flat_count = tiered->getFlatBufferIndex()->indexSize();
    out.check("tiered_training_completed",flat_count == 0 && graph->indexSize() == a.vectors);
    out.memory << scope << "\tpost_training_flat_vector_count\t" << flat_count << '\n';
    for (size_t i = 0; i < std::min(size_t(64),a.vectors); ++i) {
        const size_t id = i * (a.vectors - 1) / 63;
        auto encoded = codec.encode(data.row(id));
        require(std::memcmp(encoded.get(),recordByLabel(graph,id),codec.stored_bytes) == 0,
                "Public/diagnostic encoded record parity failed");
        auto other = codec.encode(data.row((id + 1) % a.vectors));
        const double public_pair = graph->calcDistance(recordByLabel(graph,id),
            recordByLabel(graph,(id + 1) % a.vectors));
        const double diagnostic_pair = codec.pair(encoded.get(),other.get());
        auto prepared = codec.prepare(data.query(i % a.queries));
        const double public_query = VecSimIndex_GetDistanceFrom_Unsafe(graph,id,data.query(i % a.queries));
        const double diagnostic_query = codec.score(encoded.get(),prepared.get());
        require(std::abs(public_pair - diagnostic_pair) <= 2e-6 &&
                std::abs(public_query - diagnostic_query) <= 2e-6,
                "Public/diagnostic query or graph score parity failed");
    }
    out.check("public_diagnostic_query_and_graph_score_parity",true,"64 spread records");
    out.check("public_diagnostic_encoded_parity",true,"64 spread records");
    // Observe a separate warmed query with libc-backed VecSim hooks. Hook bookkeeping
    // uses ordinary std allocation and is excluded. Restore hooks before every timing.
    timedQuery(owner.get(),data.query(0),80);
    {
        TransientMemoryProbe probe;
        timedQuery(owner.get(),data.query(0),80);
        probe.restore();
        out.check("query_transient_probe_complete",probe.complete());
        out.memory << scope << "\tquery_ef80_transient_requested_peak\t" << probe.peak()
                   << '\n' << scope << "\tquery_ef80_transient_requested_live_end\t" << probe.live()
                   << '\n' << scope << "\tquery_ef80_transient_allocations_live_end\t" << probe.liveAllocations() << '\n';
    }
    out.config << "query_transient_probe_scope\tone_separate_warmed_public_query_ef80_through_reply_cleanup_hooks_restored_before_timing\n"
               << "query_transient_probe_exclusions\tnon_VecSim_allocations_system_allocator_overhead_and_bookkeeping_excluded\n";
    for (size_t ef : {40,80,160}) {
        for (size_t q = 0; q < a.queries; ++q)
            writeRanking(out,scope,ef,q,query(owner.get(),data.query(q),ef),expected[q]);
        // First pass above warms search state; memory below includes its retained allocations.
        for (size_t batch = 0; batch < 3; ++batch) {
            const auto begin = ns();
            for (size_t q = 0; q < a.queries; ++q) timedQuery(owner.get(),data.query(q),ef);
            out.time(scope,"public_TopK_prepare_search_consume_cleanup",ef,batch,a.queries,ns() - begin);
        }
    }
    out.memory << scope << "\tsteady_index_allocator_requested\t" << owner->getAllocationSize()
               << '\n' << scope << "\tstored_record_bytes\t" << graph->getStoredDataSize()
               << '\n' << scope << "\tvector_count\t" << graph->indexSize()
               << '\n' << scope << "\tcapacity\t" << graph->indexCapacity() << '\n';
    const auto before = query(graph,data.query(0),160);
    const auto path = a.output / "selected.hnsw";
    graph->saveIndex(path.string());
    Owner reload(HNSWFactory::NewIndex(path.string()),VecSimIndex_Free);
    auto *loaded = dynamic_cast<Graph *>(reload.get()); require(loaded,"Reload type mismatch");
    out.check("reload_topK_exact",query(loaded,data.query(0),160) == before);
    out.check("reload_integrity",loaded->checkIntegrity().valid_state && loaded->indexSize() == a.vectors);
    const auto original_mean = graph->getTurboQuant()->biasMean(), loaded_mean = loaded->getTurboQuant()->biasMean();
    out.check("reload_mean_exact",original_mean.size() == loaded_mean.size() &&
              std::memcmp(original_mean.data(),loaded_mean.data(),original_mean.size_bytes()) == 0);
    for (size_t i : {size_t(0),a.vectors / 2,a.vectors - 1})
        require(std::memcmp(recordByLabel(graph,i),recordByLabel(loaded,i),codec.stored_bytes) == 0,
                "Reload record mismatch");
    out.check("reload_encoded_records_exact",true);
    out.check("reload_delete",VecSimIndex_DeleteVector(reload.get(),0) == 1);
    out.check("reload_add_after_delete",VecSimIndex_AddVector(reload.get(),data.row(0),a.vectors + 1) == 1);
    out.check("reload_mutated_integrity",loaded->indexSize() == a.vectors && loaded->checkIntegrity().valid_state);
    const auto modified = query(loaded,data.query(0),160);
    for (const auto &[score,label] : modified) require(label != 0,"Deleted label returned");
    out.check("reload_mutated_query",true);
    // Tiny additionally covers direct public factory creation with an explicitly frozen mean.
    if (a.stage == "tiny") {
        config.mean = codec.mean.data();
        Owner standalone(VecSimIndex_New(&primary),VecSimIndex_Free);
        auto *direct = dynamic_cast<Graph *>(standalone.get()); require(direct,"Standalone TQ factory failed");
        for (size_t i = 0; i < 64; ++i) {
            require(VecSimIndex_AddVector(direct,data.row(i),i) == 1,"Standalone add failed");
            auto encoded = codec.encode(data.row(i));
            require(std::memcmp(encoded.get(),recordByLabel(direct,i),codec.stored_bytes) == 0,
                    "Standalone public record parity failed");
        }
        query(direct,data.query(0),160);
        out.check("standalone_public_TQ_factory_parity",direct->checkIntegrity().valid_state);
    }
}
int run(const Args &a) {
    Output out(a); Data data(a,out);
    const size_t requested = a.training == "full" ? a.vectors : number(a.training);
    const size_t training_count = std::min(requested,a.vectors);
    const auto mean = data.mean(training_count);
    out.config << "training_actual\t" << training_count << "\nmean_fnv1a64\t" << hash(mean.data(),dim * 4) << '\n';
    Codec codec(a.codec,mean);
    if (a.stage == "micro") micro(a,data,codec,out);
    else {
        const auto expected = truth(a,data,out);
        if (a.stage == "representation") representation(a,data,codec,out,expected);
        else { numerical(codec,data,out,64); codec.memory(out,a.vectors);
               publicGraph(a,data,codec,out,expected,training_count); }
    }
    out.check("stage_complete",true); out.config << "generation\tcomplete\n";
    out.config.flush();
    std::ifstream config_tsv(a.output / "config.tsv");
    std::ofstream config_csv(a.output / "config.csv");
    config_csv.exceptions(std::ios::failbit | std::ios::badbit);
    std::string line;
    while (std::getline(config_tsv,line)) {
        const auto tab = line.find('\t');
        require(tab != std::string::npos,"Invalid config TSV row");
        auto quote = [](const std::string &s) {
            std::string result = "\"";
            for (char c : s) { if (c == '"') result += '"'; result += c; }
            return result + '"';
        };
        config_csv << quote(line.substr(0,tab)) << ',' << quote(line.substr(tab + 1)) << '\n';
    }
    return 0;
}
} // namespace selected_modes
int main(int argc,char **argv) {
    try { return selected_modes::run(selected_modes::parse(argc,argv)); }
    catch (const std::exception &e) { std::cerr << "bm_selected_modes: " << e.what() << '\n'; return 1; }
}
