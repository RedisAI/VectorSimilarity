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
#include "bm_common.h"

class BM_VecSimGeneral : public benchmark::Fixture {
public:
    static size_t block_size;

protected:
    static size_t dim;
    static size_t n_vectors;

    static const char *test_vectors_file;
    static size_t n_queries;

    static size_t ref_count;

    static std::vector<VecSimIndex *> indices;

    static const char *hnsw_index_file;

    BM_VecSimGeneral() = default;
    virtual ~BM_VecSimGeneral();

    static void MeasureRecall(VecSimQueryResult_List hnsw_results,
                              VecSimQueryResult_List bf_results, size_t &correct);

    template <typename IndexParams>
    static inline VecSimIndex *CreateNewIndex(IndexParams &index_params) {
        VecSimParams params = CreateParams(index_params);
        return VecSimIndex_New(&params);
    }

    static inline std::string AttachRootPath(const char *file_name) {
        return std::string(getenv("ROOT")) + "/" + file_name;
    }

private:
    static inline VecSimParams CreateParams(const HNSWParams &hnsw_params) {
        VecSimParams params{.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        return params;
    }

    static inline VecSimParams CreateParams(const BFParams &bf_params) {
        VecSimParams params{.algo = VecSimAlgo_BF, .bfParams = bf_params};
        return params;
    }
};

/*
 *  Populate the given queries vector with the serialized raw vectors data in
 *  the file which is located in the given path.
 */
void load_test_vectors(const char *path, std::vector<std::vector<float>> &queries, size_t n_queries,
                       size_t dim);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);
    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < BM_VecSimIndex::n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

#define VARIABLE_BATCH_SIZE_ARGS(initial_batch_size, num_batches)                                  \
    ->Args({initial_batch_size, num_batches})->ArgNames({"batch initial size", "number of batches"})

#define BATCHES_TO_ADHOC_ARGS(step, num_batches)                                                   \
    ->Args({step, num_batches})->ArgNames({"step", "number of batches"})
