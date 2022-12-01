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

    static bool is_initialized;
    static size_t first_updatedBM_ref_count;
    BM_VecSimUpdatedIndex() {
        if (!is_initialized) {
            // Initialize the updated indexes as well, if this is the first instance.
            Initialize();
            is_initialized = true;
            first_updatedBM_ref_count = REF_COUNT;
        }
    }

    ~BM_VecSimUpdatedIndex() {
        if (REF_COUNT == first_updatedBM_ref_count) {
            VecSimIndex_Free(INDICES[VecSimAlgo_BF + updated_index_offset]);
            VecSimIndex_Free(INDICES[VecSimAlgo_HNSWLIB + updated_index_offset]);
        }
    }

private:
    static const char *updated_hnsw_index_file;
    static void Initialize();
};

template <typename index_type_t>
bool BM_VecSimUpdatedIndex<index_type_t>::is_initialized = false;

template <typename index_type_t>
size_t BM_VecSimUpdatedIndex<index_type_t>::first_updatedBM_ref_count = 0;

template <typename index_type_t>
void BM_VecSimUpdatedIndex<index_type_t>::Initialize() {

    VecSimType type = index_type_t::get_index_type();

    BFParams bf_params = {.type = type,
                          .dim = DIM,
                          .metric = VecSimMetric_Cosine,
                          .multi = IS_MULTI,
                          .initialCapacity = N_VECTORS};
    // This index will be inserted after the basic indices at indices[VecSimAlfo_BF + update_offset]
    INDICES.push_back(BM_INDEX::CreateNewIndex(bf_params));

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
    // This index will be inserted after the basic indices at indices[VecSimAlgo_HNSWLIB +
    // update_offset]
    INDICES.push_back(
        HNSWFactory::NewIndex(BM_INDEX::AttachRootPath(updated_hnsw_index_file), &params));

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
