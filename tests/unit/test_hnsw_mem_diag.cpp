/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

// Diagnostic for the per-element memory of the in-RAM HNSW index that is not vector data.
// It always prints the real struct sizes. When VECSIM_MEM_DIAG_NPY points to a float32 .npy file
// (rows x dim, C order), it also builds an HNSW index over the first VECSIM_MEM_DIAG_N rows
// (default 100000) and prints the incoming-edge and link histograms, the allocator-reported index
// memory, a per-element breakdown, and a fingerprint of the graph (for before/after comparisons).
// Optional: VECSIM_MEM_DIAG_SAVE=<path> saves the built index; VECSIM_MEM_DIAG_LOAD=<path> loads an
// index from a file instead of building one (no allocator comparison then);
// VECSIM_MEM_DIAG_QUERIES / VECSIM_MEM_DIAG_TRUTH (float32 / int32 .npy) run top-10 queries at
// ef=100 and print a fingerprint of the results and the recall.

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "unit_test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

// Minimal reader for a version 1 or 2 float32 .npy file. Returns the row count and fills dim.
template <typename T>
std::vector<T> LoadNpy(const char *path, const char *descr, size_t max_rows, size_t &rows,
                       size_t &dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not an npy file");
    }
    size_t header_len = 0;
    if (magic[6] == 1) {
        uint16_t len;
        in.read(reinterpret_cast<char *>(&len), 2);
        header_len = len;
    } else {
        uint32_t len;
        in.read(reinterpret_cast<char *>(&len), 4);
        header_len = len;
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (header.find(descr) == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error(std::string("expected a C-order npy file of ") + descr);
    }
    auto shape_pos = header.find("'shape': (");
    size_t file_rows = 0;
    std::sscanf(header.c_str() + shape_pos, "'shape': (%zu, %zu)", &file_rows, &dim);
    rows = std::min(file_rows, max_rows);
    std::vector<T> data(rows * dim);
    in.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T));
    return data;
}

uint64_t Mix(uint64_t h, uint64_t v) {
    // FNV-1a over the 8 bytes of v.
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xff;
        h *= 1099511628211ULL;
    }
    return h;
}

void PrintHistogram(const char *title, const std::map<size_t, size_t> &hist, size_t total) {
    printf("%s\n", title);
    for (auto &[k, v] : hist) {
        printf("  %4zu : %8zu  (%5.2f%%)\n", k, v, 100.0 * v / total);
    }
}

} // namespace

TEST(HNSWMemoryDiagnostic, breakdown) {
    const size_t hdr = VecSimAllocator::getAllocationOverheadSize();
    printf("sizeof(ElementGraphData)            = %zu\n", sizeof(ElementGraphData));
    printf("sizeof(ElementLevelData)            = %zu\n", sizeof(ElementLevelData));
    printf("offsetof(ElementLevelData, links)   = %zu\n", offsetof(ElementLevelData, links));
    printf("offsetof(ElementGraphData, level0)  = %zu\n", offsetof(ElementGraphData, level0));
    printf("sizeof(IncomingEdges)               = %zu\n", sizeof(IncomingEdges));
    printf("sizeof(std::vector<idType, STLAlloc>) = %zu\n",
           sizeof(std::vector<idType, VecsimSTLAllocator<idType>>));
    printf("sizeof(VecsimBaseObject)            = %zu\n", sizeof(VecsimBaseObject));
    printf("sizeof(ElementMetaData)             = %zu\n", sizeof(ElementMetaData));
    printf("sizeof(one_byte_mutex)              = %zu\n", sizeof(vecsim_stl::one_byte_mutex));
    printf("sizeof(tag_t)                       = %zu\n", sizeof(tag_t));
    printf("allocator header                    = %zu\n", hdr);

    const char *npy = std::getenv("VECSIM_MEM_DIAG_NPY");
    if (!npy) {
        GTEST_SKIP() << "set VECSIM_MEM_DIAG_NPY to a float32 .npy file for the index part";
    }
    size_t max_rows = 100000;
    if (const char *n_env = std::getenv("VECSIM_MEM_DIAG_N")) {
        max_rows = std::strtoull(n_env, nullptr, 10);
    }
    size_t n = 0, dim = 0;
    std::vector<float> data = LoadNpy<float>(npy, "'<f4'", max_rows, n, dim);
    printf("data: %zu rows, dim %zu\n", n, dim);
    const char *load_path = std::getenv("VECSIM_MEM_DIAG_LOAD");

    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .dim = dim,
                         .metric = VecSimMetric_IP,
                         .multi = false,
                         .M = 16,
                         .efConstruction = 100,
                         .efRuntime = 100};
    VecSimParams vecsim_params = CreateParams(params);
    VecSimIndex *index =
        load_path ? HNSWFactory::NewIndex(load_path) : VecSimIndex_New(&vecsim_params);
    ASSERT_NE(index, nullptr);
    auto *hnsw = dynamic_cast<HNSWIndex_Single<float, float> *>(index);
    ASSERT_NE(hnsw, nullptr);

    const size_t mem_empty = load_path ? 0 : index->getAllocationSize();
    if (load_path) {
        hnsw->setEf(100);
        printf("loaded %s\n", load_path);
    } else {
        for (size_t i = 0; i < n; i++) {
            VecSimIndex_AddVector(index, data.data() + i * dim, i);
        }
    }
    const size_t mem_full = index->getAllocationSize();
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_TRUE(hnsw->checkIntegrity().valid_state);
    if (const char *save_path = std::getenv("VECSIM_MEM_DIAG_SAVE")) {
        hnsw->saveIndex(save_path);
        printf("saved %s\n", save_path);
    }

    if (const char *q_path = std::getenv("VECSIM_MEM_DIAG_QUERIES")) {
        size_t nq = 0, qdim = 0;
        std::vector<float> queries = LoadNpy<float>(q_path, "'<f4'", SIZE_MAX, nq, qdim);
        ASSERT_EQ(qdim, dim);
        std::vector<int32_t> truth;
        size_t tk = 0, tn = 0;
        if (const char *t_path = std::getenv("VECSIM_MEM_DIAG_TRUTH")) {
            truth = LoadNpy<int32_t>(t_path, "'<i4'", SIZE_MAX, tn, tk);
        }
        const size_t k = 10;
        uint64_t results_fp = 1469598103934665603ULL;
        size_t hits = 0;
        for (size_t q = 0; q < nq; q++) {
            VecSimQueryReply *reply =
                VecSimIndex_TopKQuery(index, queries.data() + q * dim, k, nullptr, BY_SCORE);
            VecSimQueryReply_Iterator *it = VecSimQueryReply_GetIterator(reply);
            while (VecSimQueryResult *r = VecSimQueryReply_IteratorNext(it)) {
                int64_t id = VecSimQueryResult_GetId(r);
                results_fp = Mix(results_fp, (uint64_t)id);
                if (!truth.empty()) {
                    hits += std::find(truth.begin() + q * tk, truth.begin() + q * tk + k, id) !=
                            truth.begin() + q * tk + k;
                }
            }
            VecSimQueryReply_IteratorFree(it);
            VecSimQueryReply_Free(reply);
        }
        printf("queries: %zu, results fingerprint %016llx, recall@10 %.4f\n", nq,
               (unsigned long long)results_fp, truth.empty() ? 0.0 : (double)hits / (nq * k));
    }

    // Walk the graph.
    std::map<size_t, size_t> links0_hist, in0_size_hist, in0_cap_hist, level_hist;
    size_t in_objects = 0, in_nonempty = 0, in_elems = 0, in_cap = 0;
    size_t in0_nonempty = 0, in0_elems = 0, in0_cap = 0;
    size_t upper_levels = 0, upper_alloc = 0, links0_total = 0;
    uint64_t links_fp = 1469598103934665603ULL, incoming_fp = 1469598103934665603ULL;
    for (idType id = 0; id < n; id++) {
        ElementGraphData *g = hnsw->getGraphDataByInternalId(id);
        level_hist[g->toplevel]++;
        if (g->toplevel > 0) {
            upper_levels += g->toplevel;
            upper_alloc += 1;
        }
        links_fp = Mix(links_fp, g->toplevel);
        for (size_t l = 0; l <= g->toplevel; l++) {
            ElementLevelData &ld = hnsw->getElementLevelData(g, l);
            links_fp = Mix(links_fp, ld.getNumLinks());
            for (size_t j = 0; j < ld.getNumLinks(); j++) {
                links_fp = Mix(links_fp, ld.getLinkAtPos(j));
            }
            auto in = ld.getIncomingEdges();
            const size_t in_capacity = ld.getIncomingEdgesCapacity();
            std::vector<idType> sorted_in(in.begin(), in.end());
            std::sort(sorted_in.begin(), sorted_in.end());
            incoming_fp = Mix(incoming_fp, sorted_in.size());
            for (idType x : sorted_in) {
                incoming_fp = Mix(incoming_fp, x);
            }
            in_objects++;
            in_elems += in.size();
            in_cap += in_capacity;
            in_nonempty += in_capacity > 0;
            if (l == 0) {
                links0_hist[ld.getNumLinks()]++;
                links0_total += ld.getNumLinks();
                in0_size_hist[in.size()]++;
                in0_cap_hist[in_capacity]++;
                in0_elems += in.size();
                in0_cap += in_capacity;
                in0_nonempty += in_capacity > 0;
            }
        }
    }

    printf("\nindex: n=%zu, M=%zu, M0=%zu, elementGraphDataSize=%zu, levelDataSize=%zu\n", n,
           hnsw->M, hnsw->M0, hnsw->elementGraphDataSize, hnsw->levelDataSize);
    printf("graph fingerprint: links=%016llx incoming=%016llx\n", (unsigned long long)links_fp,
           (unsigned long long)incoming_fp);
    PrintHistogram("toplevel histogram:", level_hist, n);
    PrintHistogram("level-0 link count histogram:", links0_hist, n);
    printf("level-0 mean links: %.2f\n", (double)links0_total / n);
    PrintHistogram("level-0 incoming-edge list size histogram:", in0_size_hist, n);
    PrintHistogram("level-0 incoming-edge list capacity histogram:", in0_cap_hist, n);
    printf("level-0 incoming lists: %zu, with a buffer %zu (%.2f%%), elements %zu, capacity %zu\n",
           n, in0_nonempty, 100.0 * in0_nonempty / n, in0_elems, in0_cap);
    printf("all-level incoming lists: %zu, with a buffer %zu, elements %zu, capacity %zu\n",
           in_objects, in_nonempty, in_elems, in_cap);

    // Per-element breakdown from the structures, compared with the allocator.
    const size_t blocks = hnsw->graphDataBlocks.size();
    const size_t cap = hnsw->maxElements;
    const double N = (double)n;
    struct Row {
        const char *name;
        double bytes;
    };
    std::vector<Row> rows = {
        {"graph record blocks (capacity x elementGraphDataSize)",
         (double)cap * hnsw->elementGraphDataSize + blocks * hdr},
        {"incoming-edge lists (header + ids + allocator header)",
         (double)in_nonempty * (sizeof(IncomingEdges) + hdr) + in_cap * sizeof(idType)},
        {"upper-level records", (double)upper_levels * hnsw->levelDataSize + upper_alloc * hdr},
        {"vector blocks (capacity x storedDataSize)",
         (double)cap * hnsw->getStoredDataSize() + blocks * hdr},
        {"idToMetaData (capacity)",
         (double)hnsw->idToMetaData.capacity() * sizeof(ElementMetaData)},
        {"elementLocks (capacity)",
         (double)hnsw->elementLocks.capacity() * sizeof(vecsim_stl::one_byte_mutex)},
    };
    // Label map: rebuild an identical map on a fresh allocator and measure it.
    {
        auto alloc = VecSimAllocator::newVecsimAllocator();
        size_t before = alloc->getAllocationSize();
        {
            vecsim_stl::unordered_map<labelType, idType> m(alloc);
            m.reserve(cap);
            for (size_t i = 0; i < n; i++) {
                m[i] = i;
            }
            printf("label map: size %zu, bucket_count %zu (index map bucket_count %zu)\n", m.size(),
                   m.bucket_count(), hnsw->labelLookup.bucket_count());
            rows.push_back({"label map (replica)", (double)(alloc->getAllocationSize() - before)});
        }
    }
    double sum = 0;
    printf("\n%-55s %12s %10s\n", "component", "bytes", "B/elem");
    for (auto &r : rows) {
        printf("%-55s %12.0f %10.2f\n", r.name, r.bytes, r.bytes / N);
        sum += r.bytes;
    }
    const double delta = (double)mem_full - mem_empty;
    printf("%-55s %12.0f %10.2f\n", "sum of components", sum, sum / N);
    printf("%-55s %12.0f %10.2f\n", "allocator: full - empty", delta, delta / N);
    printf("%-55s %12zu %10.2f\n", "allocator: full", mem_full, mem_full / N);
    printf("%-55s %12.0f %10.2f\n", "unexplained (full - empty - sum)", delta - sum,
           (delta - sum) / N);

    VecSimIndex_Free(index);
}
