#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/algorithms/hnsw/serialization.h"
#include "VecSim/algorithms/brute_force/brute_force.h"

class BM_VecSimBasics : public benchmark::Fixture {
protected:
    static VecSimIndex *bf_index;
    static VecSimIndex *hnsw_index;
    static size_t dim;
    static size_t n_vectors;
    static std::vector<std::vector<float>> *queries;
    static size_t n_queries;

    // We use this class as a singleton for every test case, so we won't hold several indices (to
    // reduce memory consumption).
    static size_t ref_count;

    BM_VecSimBasics();

public:
    static void Initialize(size_t M, size_t ef_c, const char *hnsw_index_path,
                           const char *test_vectors_path);
    static void RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                             size_t &correct, VecSimIndex *hnsw_index_, VecSimIndex *bf_index_);
    virtual ~BM_VecSimBasics();
};

/*
 *  Populate the given hnsw_index with the serialized index data in the file
 *  which is located in the given path.
 */
void load_HNSW_index(const char *path, VecSimIndex *hnsw_index);

/*
 *  Populate the given queries vector with the serialized raw vectors data in
 *  the file which is located in the given path.
 */
void load_test_vectors(const char *path, std::vector<std::vector<float>> &queries, size_t n_queries,
                       size_t dim);
