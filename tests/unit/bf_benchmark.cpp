#include <random>
#include <iostream>
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

    void get_10000_total_results(size_t res_per_iter) {
        std::vector<float> query(dim);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < dim; ++i) {
            query[i] = (float)distrib(rng);
        }
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(bf_index, query.data());
        size_t res_num = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, res_per_iter, BY_SCORE);
            res_num += VecSimQueryResult_Len(res);
            if (res_num == 1000) {
                break;
            }
        }
    }

    void TearDown() {
        VecSimIndex_Free(bf_index);
    }
};

int main() {
    BM_BatchIteratorBF bm;
    bm.SetUp();
    for (size_t i=0; i<100; i++) {
        bm.get_10000_total_results(1000);
    }
    bm.TearDown();
}
