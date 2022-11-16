/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include <istream>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_interface.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "bm_definitions.h"

// This class includes every static data member that is:
// 1. common for fp32 and fp64 data sets.
// or 2. in use for all types of bench mark, if this type
// is defined in a separate compilation unit.
class BM_VecSimGeneral : public benchmark::Fixture {
public:
    static size_t block_size;

protected:
    static size_t dim;
    static size_t M;
    static size_t EF_C;
    static size_t n_vectors;

    static bool is_multi;
    static size_t n_queries;
    static const char *hnsw_index_file;
    static const char *test_queries_file;
    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral() = default;

    static void MeasureRecall(VecSimQueryResult_List hnsw_results,
                              VecSimQueryResult_List bf_results, size_t &correct);

protected:
    static inline VecSimQueryParams CreateQueryParams(const HNSWRuntimeParams &RuntimeParams) {
        VecSimQueryParams QueryParams = {.hnswRuntimeParams = RuntimeParams};
        return QueryParams;
    }

    static inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        return params;
    }

    static inline VecSimParams CreateParams(const BFParams &bf_params) {
        VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
        return params;
    }
};
