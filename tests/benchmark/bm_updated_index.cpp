/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/serializer.h"
#include "bm_basics.h"

// Global benchmark data
size_t BM_VecSimGeneral::n_vectors = 500000;

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512.hnsw";

const char *updated_hnsw_index_file =
    "tests/benchmark/data/DBpedia-n500K-cosine-d768-M65-EFC512-updated.hnsw";
template <typename index_type_t>
class BM_VecSimUpdatedIndex : public BM_VecSimBasics<index_type_t> {
public:
    const static Offset_t updated_index_offset = 2;
    using BM_INDEX = BM_VecSimIndex<index_type_t>;

    BM_VecSimUpdatedIndex() {
        if (BM_VecSimGeneral::ref_count == 1) {
            // Initialize the updated indexes as well, if this is the first instance.
            Initialize();
        }
    }

    void Initialize();

public:
    ~BM_VecSimUpdatedIndex() {
        if (BM_VecSimGeneral::ref_count == 1) {
            VecSimIndex_Free(INDICES[VecSimAlgo_BF + updated_index_offset]);
            VecSimIndex_Free(INDICES[VecSimAlgo_HNSWLIB + updated_index_offset]);
        }
    }
};

template <typename index_type_t>
void BM_VecSimUpdatedIndex<index_type_t>::Initialize() {
    BFParams bf_params = {.type = index_type_t::get_index_type(),
                          .dim = DIM,
                          .metric = VecSimMetric_Cosine,
                          .initialCapacity = N_VECTORS};
    // this index will be inserted after the basic indices at indices[VecSimAlfo_BF + update_offset]
    INDICES.push_back(BM_VecSimGeneral::CreateNewIndex(bf_params));

    // Initially, load all the vectors to the updated bf index (before we override it).
    for (size_t i = 0; i < N_VECTORS; ++i) {
        char *blob = BM_INDEX::GetHNSWDataByInternalId(i);
        size_t label = BM_INDEX::CastToHNSW(INDICES[VecSimAlgo_HNSWLIB])->getExternalLabel(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + updated_index_offset], blob, label);
    }

    // Initialize and populate an HNSW index where all vectors have been updated.
    HNSWParams hnsw_params = {.type = index_type_t::get_index_type(),
                              .dim = DIM,
                              .metric = VecSimMetric_Cosine,
                              .initialCapacity = N_VECTORS};
    INDICES.push_back(BM_VecSimGeneral::CreateNewIndex(hnsw_params));

    // Load pre-generated HNSW index. Index file path is relative to repository root dir.
    BM_INDEX::LoadHNSWIndex(BM_VecSimGeneral::AttachRootPath(updated_hnsw_index_file),
                            updated_index_offset);

    if (!BM_INDEX::CastToHNSW(INDICES[VecSimAlgo_HNSWLIB + updated_index_offset])
             ->serializingIsValid()) {
        throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
    }
    // Add the same vectors to the *updated* FLAT index (override the previous vectors).
    for (size_t i = 0; i < N_VECTORS; ++i) {
        char *blob = BM_INDEX::GetHNSWDataByInternalId(i, updated_index_offset);
        size_t label = BM_INDEX::CastToHNSW(INDICES[VecSimAlgo_HNSWLIB + updated_index_offset])
                           ->getExternalLabel(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + updated_index_offset], blob, label);
    }
}

// TopK search BM
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st, updated_index_offset);
}
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t k = st.range(0);
    TopK_BF(k, st, updated_index_offset);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st, updated_index_offset);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t ef = st.range(0);
    size_t k = st.range(1);
    TopK_HNSW(ef, k, st, updated_index_offset);
}

// Index memory metrics - run only once.
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated_fp32, fp32_index_t)
(benchmark::State &st) { Memory_FLAT(st, updated_index_offset); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated_fp64, fp64_index_t)
(benchmark::State &st) { Memory_FLAT(st, updated_index_offset); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated_fp32, fp32_index_t)
(benchmark::State &st) { Memory_HNSW(st, updated_index_offset); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated_fp64, fp64_index_t)
(benchmark::State &st) { Memory_HNSW(st, updated_index_offset); }

BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_FLAT_updated_fp64)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated_fp32)->Iterations(1);
BENCHMARK_REGISTER_F(BM_VecSimUpdatedIndex, Memory_HNSW_updated_fp64)->Iterations(1);

REGISTER_TopK_BF(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp32);
REGISTER_TopK_BF(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp64);

REGISTER_TopK_HNSW(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp32);
REGISTER_TopK_HNSW(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp64);

BENCHMARK_MAIN();
