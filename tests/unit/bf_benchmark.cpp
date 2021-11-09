#include <random>
#include <iostream>
#include <chrono>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"

class BM_BatchIteratorBF {
protected:
    std::mt19937 rng;
    VecSimIndex *bf_index;
    size_t dim;

public:
    void SetUp() {
        // Initialize BF index
        dim = 100;
        size_t n_vectors = 1000000;
        VecSimParams params = {.bfParams = {.initialCapacity = n_vectors},
                .type = VecSimType_FLOAT32,
                .size = dim,
                .metric = VecSimMetric_L2,
                .algo = VecSimAlgo_BF};
        bf_index = VecSimIndex_New(&params);

        // Add 1M random vectors with dim=100
        std::vector<float> data(n_vectors * dim);
        rng.seed(47);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < n_vectors * dim; ++i) {
            data[i] = (float)distrib(rng);
        }
        for (size_t i = 0; i < n_vectors; ++i) {
            VecSimIndex_AddVector(bf_index, data.data() + dim*i, i);
        }
        std::cout << "set-up is done" << std::endl;
    }

    long long get_10000_total_results(size_t res_per_iter) {
        long long search_time = 0;
        std::vector<float> query(dim);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(bf_index, query.data());
        size_t res_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            auto start = std::chrono::high_resolution_clock::now();
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, res_per_iter, BY_SCORE);
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            search_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            res_num += VecSimQueryResult_Len(res);
            if (res_num == 10000) {
                break;
            }
        }
        return search_time;
    }

    void TearDown() {
        VecSimIndex_Free(bf_index);
    }
};

int main() {
    BM_BatchIteratorBF bm;
    bm.SetUp();
    long long total_time = 0;
    size_t res_per_iter = 100;
    size_t n = 100;
    for (size_t i=0; i<n; i++) {
        total_time += bm.get_10000_total_results(res_per_iter);
    }
    std::cout << "Avg time for " << res_per_iter << " results per iteration is: " << (double) total_time/n/1000000
    << " seconds" << std::endl;
    bm.TearDown();
}
