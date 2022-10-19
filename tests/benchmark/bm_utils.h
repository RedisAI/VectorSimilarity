#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/vec_sim_interface.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"

class BM_VecSimBasics : public benchmark::Fixture {
public:
    static VecSimIndex *bf_index;
    static HNSWIndex<float, float> *hnsw_index;
    static size_t dim;
    static size_t n_vectors;
    static size_t M;
    static size_t EF_C;
    static size_t block_size;

    static const char *test_vectors_file;
    static std::vector<std::vector<float>> *queries;
    static size_t n_queries;

    // We use this class as a singleton for every test case, so we won't hold several indices (to
    // reduce memory consumption).
    static size_t ref_count;

    BM_VecSimBasics();

    static void Initialize();
    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             size_t &correct, VecSimIndex *hnsw_index_, VecSimIndex *bf_index_);
    virtual ~BM_VecSimBasics();

    static const char *hnsw_index_file;
    static inline std::string GetSerializedIndexLocation(const char *file_name) {
        return std::string(getenv("ROOT")) + "/" + file_name;
    }
};

/*
 *  Populate the given queries vector with the serialized raw vectors data in
 *  the file which is located in the given path.
 */
void load_test_vectors(const char *path, std::vector<std::vector<float>> &queries, size_t n_queries,
                       size_t dim);

#define HNSW_TOP_K_ARGS(ef_runtime, k) ->Args({ef_runtime, k})->ArgNames({"ef_runtime", "k"})

#define FIXED_BATCH_SIZE_ARGS(batch_size, num_batches)                                             \
    ->Args({batch_size, num_batches})->ArgNames({"batch size", "number of batches"})

#define VARIABLE_BATCH_SIZE_ARGS(initial_batch_size, num_batches)                                  \
    ->Args({initial_batch_size, num_batches})->ArgNames({"batch initial size", "number of batches"})

#define BATCHES_TO_ADHOC_ARGS(step, num_batches)                                                   \
    ->Args({step, num_batches})->ArgNames({"step", "number of batches"})
