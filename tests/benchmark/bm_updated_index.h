#pragma once

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "bm_common.h"

/**************************************
  Basic tests for updated single value index.
***************************************/

template <typename index_type_t>
class BM_VecSimUpdatedIndex : public BM_VecSimCommon<index_type_t> {
public:
    const static Offset_t updated_index_offset = 2;
    using BM_INDEX = BM_VecSimIndex<index_type_t>;

    static std::vector<const char *> updated_hnsw_index_files;

    BM_VecSimUpdatedIndex() {
        if (BM_VecSimGeneral::ref_count == 1) {
            // Initialize the updated indexes as well, if this is the first instance.
            Initialize();
        }
    }

    ~BM_VecSimUpdatedIndex() {
        if (BM_VecSimGeneral::ref_count == 1) {
            VecSimIndex_Free(INDICES[VecSimAlgo_BF + updated_index_offset]);
            VecSimIndex_Free(INDICES[VecSimAlgo_HNSWLIB + updated_index_offset]);
        }
    }

private:
    static void Initialize();
};

template <typename index_type_t>
void BM_VecSimUpdatedIndex<index_type_t>::Initialize() {

    VecSimType type = index_type_t::get_index_type();

    BFParams bf_params = {.type = type,
                          .dim = DIM,
                          .metric = VecSimMetric_Cosine,
                          .multi = IS_MULTI,
                          .initialCapacity = N_VECTORS};
    // This index will be inserted after the basic indices at indices[VecSimAlfo_BF + update_offset]
    INDICES.push_back(BM_VecSimGeneral::CreateNewIndex(bf_params));

    // Initially, load all the vectors to the updated bf index (before we override it).
    for (size_t i = 0; i < N_VECTORS; ++i) {
        char *blob = BM_INDEX::GetHNSWDataByInternalId(i);
        size_t label = BM_INDEX::CastToHNSW(INDICES[VecSimAlgo_HNSWLIB])->getExternalLabel(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + updated_index_offset], blob, label);
    }

    // Generate the updated index from file.
    // HNSWParams is required to load v1 index
    HNSWParams params = {.type = type,
                         .dim = DIM,
                         .metric = VecSimMetric_Cosine,
                         .multi = IS_MULTI,
                         .blockSize = BM_VecSimGeneral::block_size};

    // Generate index from file.
    // This index will be inserted after the basic indices at indices[VecSimAlfo_HNSWLIB +
    // update_offset]
    INDICES.push_back(HNSWFactory::NewIndex(
        BM_VecSimGeneral::AttachRootPath(updated_hnsw_index_files[type]), &params));

    if (!BM_INDEX::CastToHNSW(INDICES[VecSimAlgo_HNSWLIB + updated_index_offset])
             ->checkIntegrity()
             .valid_state) {
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
(benchmark::State &st) { TopK_BF(st, updated_index_offset); }
BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_BF_Updated_fp64, fp64_index_t)
(benchmark::State &st) { TopK_BF(st, updated_index_offset); }

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp32, fp32_index_t)
(benchmark::State &st) { TopK_HNSW(st, updated_index_offset); }

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimUpdatedIndex, TopK_HNSW_Updated_fp64, fp64_index_t)
(benchmark::State &st) { TopK_HNSW(st, updated_index_offset); }

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
