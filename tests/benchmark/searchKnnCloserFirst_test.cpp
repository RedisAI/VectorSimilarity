// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include <assert.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <sys/time.h>
#include <ctime>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "VecSim/utils/arr_cpp.h"
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::system_clock;

namespace
{

using idx_t = size_t;

void test() {
    size_t d = 16;
    idx_t n = 100000;
    idx_t nq = 10;
    size_t k = 10;
    size_t itr = 100000;
   
    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }
    
    VecSimParams params{.algo = VecSimAlgo_HNSWLIB,
                        .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                 .dim = d,
                                                 .metric = VecSimMetric_L2,
                                                 .initialCapacity = n,
                                                 .M = 16,
                                                 .efConstruction = 200}};
    VecSimIndex *index = VecSimIndex_New(&params);

    for (size_t i = 0; i < n; i++) {
        VecSimIndex_AddVector(index, data.data() + d * i, i);
    }

    std::cout << "Ready for benchmark" << std::endl;
    double qps = 0;
    for (size_t i = 0; i < itr; i++) {
    // while (true) {
        auto s = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        for (size_t j = 0; j < nq; ++j) {
            const void* p = query.data() + j * d;
            auto r = VecSimIndex_TopKQuery(index, p, k, NULL, BY_SCORE);
            VecSimQueryResult_Free(r);
        }
        auto e = duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
        // std::cout << "qps\t" << ((double)nq*1000000000/(e-s)) << std::endl;
        qps += ((double)nq/(e-s));
    }
    std::cout << "qps\t" << qps*1000000000/itr << std::endl;
}

} // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}
