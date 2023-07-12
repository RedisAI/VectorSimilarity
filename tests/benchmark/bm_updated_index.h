#pragma once

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/serializer.h"
#include "VecSim/index_factories/hnsw_factory.h"
#include "bm_common.h"

/**************************************
  Basic tests for updated single value index.
***************************************/

template <typename index_type_t>
class BM_VecSimUpdatedIndex : public BM_VecSimCommon<index_type_t> {
public:
    const static unsigned short updated_index_offset = 3;
    // The constructor is called after we have already registered the tests residing in
    // BM_VecSimCommon, (and not in this class) so `ref_count` is not zero at the first time
    // BM_VecSimUpdatedIndex Ctor is called, and we can't rely on it to decide whether we should
    // initialize the indices or not. This is why we use the `is_initialized` flag. Also, we keep
    // the value of ref_count at the moment of initialization in first_updatedBM_ref_count to free
    // the indices when ref_count is decreased to this value. Reminder: ref_count is updated in
    // BM_VecSimIndex ctor (and dtor).
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
    INDICES.push_back(BM_VecSimIndex<index_type_t>::CreateNewIndex(bf_params));

    // Initially, load all the vectors to the updated bf index (before we override it).
    for (size_t i = 0; i < N_VECTORS; ++i) {
        const char *blob = BM_VecSimIndex<index_type_t>::GetHNSWDataByInternalId(i);
        size_t label = BM_VecSimIndex<index_type_t>::CastToHNSW(INDICES[VecSimAlgo_HNSWLIB])
                           ->getExternalLabel(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + updated_index_offset], blob, label);
    }

    // Generate index from file.
    // This index will be inserted after the basic indices at indices[VecSimAlgo_HNSWLIB +
    // update_offset]
    INDICES.push_back(HNSWFactory::NewIndex(
        BM_VecSimIndex<index_type_t>::AttachRootPath(updated_hnsw_index_file)));

    if (!BM_VecSimIndex<index_type_t>::CastToHNSW(
             INDICES[VecSimAlgo_HNSWLIB + updated_index_offset])
             ->checkIntegrity()
             .valid_state) {
        throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
    }
    // Add the same vectors to the *updated* FLAT index (override the previous vectors).
    for (size_t i = 0; i < N_VECTORS; ++i) {
        const char *blob =
            BM_VecSimIndex<index_type_t>::GetHNSWDataByInternalId(i, updated_index_offset);
        size_t label = BM_VecSimIndex<index_type_t>::CastToHNSW(
                           INDICES[VecSimAlgo_HNSWLIB + updated_index_offset])
                           ->getExternalLabel(i);
        VecSimIndex_AddVector(INDICES[VecSimAlgo_BF + updated_index_offset], blob, label);
    }
}
