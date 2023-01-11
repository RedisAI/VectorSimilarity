/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "gtest/gtest.h"
#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "test_utils.h"
#include "VecSim/query_result_struct.h"
#include <unistd.h>
#include <random>
#include <thread>
#include <atomic>

template <typename index_type_t>
class HNSWTestParallel : public ::testing::Test {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

protected:
    VecSimIndex *CreateNewIndex(HNSWParams &params, bool is_multi = false) {
        return test_utils::CreateNewIndex(params, index_type_t::get_index_type(), is_multi);
    }
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    HNSWIndex_Single<data_t, dist_t> *CastToHNSW_Single(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex_Single<data_t, dist_t> *>(index);
    }
};

// DataTypeSet, TEST_DATA_T and TEST_DIST_T are defined in test_utils.h

TYPED_TEST_SUITE(HNSWTestParallel, DataTypeSet);

TYPED_TEST(HNSWTestParallel, parallelSearchKnn) {
    size_t n = 20000;
    size_t k = 11;
    size_t dim = 45;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 64,
                         .efConstruction = 200,
                         .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t n_threads = MIN(8, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Run parallel searches where every searching thread expects to get different labels as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            size_t diff_id = (id > query_val) ? (id - query_val) : (query_val - id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    size_t memory_before = index->info().hnswInfo.memory;
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);

    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);
    // Make sure that we properly update the allocator atomically during the searches. The expected
    // Memory delta should only be the visited nodes handler added to the pool.
    size_t expected_memory = memory_before + (index->info().hnswInfo.visitedNodesPoolSize - 1) *
                                                 (sizeof(VisitedNodesHandler) + sizeof(tag_t) * n +
                                                  2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->info().hnswInfo.memory);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTestParallel, parallelSearchKNNMulti) {
    size_t dim = 45;
    size_t n = 20000;
    size_t n_labels = 1000;
    size_t k = 11;

    HNSWParams params = {
        .dim = dim, .metric = VecSimMetric_L2, .initialCapacity = n, .M = 64, .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params, true);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i % n_labels, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);
    ASSERT_EQ(index->indexLabelCount(), n_labels);

    size_t n_threads = MIN(8, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    // Run parallel searches where every searching thread expects to get different label as results
    // (determined by the thread id), which are labels in the range [50+myID-5, 50+myID+5].
    auto parallel_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            size_t diff_id = (id > query_val) ? (id - query_val) : (query_val - id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * ((res_index + 1) / 2) * ((res_index + 1) / 2)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_search, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);

    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTestParallel, parallelSearchCombined) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 64;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 64,
                         .efConstruction = 200,
                         .efRuntime = n};
    VecSimIndex *index = this->CreateNewIndex(params);

    for (size_t i = 0; i < n; i++) {
        GenerateAndAddVector<TEST_DATA_T>(index, dim, i, i);
    }
    ASSERT_EQ(VecSimIndex_IndexSize(index), n);

    size_t n_threads = MIN(15, std::thread::hardware_concurrency());
    std::atomic_int successful_searches(0);
    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    /* Run parallel searches of three kinds: KNN, range, and batched search. */

    // In knn, we expect to get different labels as results (determined by the thread id), which are
    // labels in the range [50+myID-5, 50+myID+5].
    auto parallel_knn_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query_val = 50 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, query_val);
        auto verify_res = [&](size_t id, double score, size_t res_index) {
            // We expect to get the results with increasing order of the distance between the res
            // label and the query val (query_val, query_val-1, query_val+1, query_val-2,
            // query_val+2, ...) The score is the L2 distance between the vectors that correspond
            // the ids.
            size_t diff_id = std::abs(id - query_val);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, (dim * (diff_id * diff_id)));
        };
        runTopKSearchTest(index, query, k, verify_res);
        successful_searches++;
    };

    auto parallel_range_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T pivot_id = 100 + myID;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, pivot_id);
        auto verify_res_by_score = [&](size_t id, double score, size_t res_index) {
            size_t diff_id = std::abs(id - pivot_id);
            ASSERT_EQ(diff_id, (res_index + 1) / 2);
            ASSERT_EQ(score, dim * (diff_id * diff_id));
        };
        uint expected_num_results = 11;
        // To get 11 results in the range [pivot_id-5, pivot_id+5], set the radius as the L2 score
        // in the boundaries.
        double radius = (double)dim * pow((double)expected_num_results / 2, 2);
        runRangeQueryTest(index, query, radius, verify_res_by_score, expected_num_results,
                          BY_SCORE);
        successful_searches++;
    };

    auto parallel_batched_search = [&](int myID) {
        completed_tasks[myID]++;
        TEST_DATA_T query[dim];
        GenerateVector<TEST_DATA_T>(query, dim, n);

        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(index, query, nullptr);
        size_t iteration_num = 0;

        // Get the 5 vectors whose ids are the maximal among those that hasn't been returned yet
        // in every iteration. The results order should be sorted by their score (distance from the
        // query vector), which means sorted from the largest id to the lowest.
        // Run different number of iterations for every thread id.
        size_t total_iterations = myID;
        size_t n_res = 5;
        while (VecSimBatchIterator_HasNext(batchIterator) && iteration_num < total_iterations) {
            std::vector<size_t> expected_ids(n_res);
            for (size_t i = 0; i < n_res; i++) {
                expected_ids[i] = (n - iteration_num * n_res - i - 1);
            }
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                ASSERT_TRUE(expected_ids[res_index] == id);
            };
            runBatchIteratorSearchTest(batchIterator, n_res, verify_res);
            iteration_num++;
        }
        ASSERT_EQ(iteration_num, total_iterations);
        VecSimBatchIterator_Free(batchIterator);
        successful_searches++;
    };

    std::thread thread_objs[n_threads];
    size_t memory_before = index->info().hnswInfo.memory;
    for (size_t i = 0; i < n_threads; i++) {
        if (i % 3 == 0) {
            thread_objs[i] = std::thread(parallel_knn_search, i);
        } else if (i % 3 == 1) {
            thread_objs[i] = std::thread(parallel_range_search, i);
        } else {
            thread_objs[i] = std::thread(parallel_batched_search, i);
        }
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(successful_searches, n_threads);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), 1);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()), 1);

    // Make sure that we properly update the allocator atomically during the searches.
    // Memory delta should only be the visited nodes handler added to the pool.
    size_t expected_memory = memory_before + (index->info().hnswInfo.visitedNodesPoolSize - 1) *
                                                 (sizeof(VisitedNodesHandler) + sizeof(tag_t) * n +
                                                  2 * sizeof(size_t) + sizeof(void *));
    ASSERT_EQ(expected_memory, index->info().hnswInfo.memory);
    VecSimIndex_Free(index);
}

TYPED_TEST(HNSWTestParallel, parallelInsert) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *parallel_index = this->CreateNewIndex(params);
    size_t n_threads = 10;

    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);

    auto parallel_insert = [&](int myID) {
        for (labelType label = myID; label < n; label += n_threads) {
            completed_tasks[myID]++;
            GenerateAndAddVector<TEST_DATA_T>(parallel_index, dim, label, label);
        }
    };
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), n / n_threads);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceil((double)n / n_threads));

    TEST_DATA_T query[dim];
    GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)n / 2);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the res
        // label and the query val (n/2, n/2-1, n/2+1, n/2-2, n/2+2, ...) The score is the L2
        // distance between the vectors that correspond the ids.
        size_t diff_id = std::abs(int(id - n / 2));
        ASSERT_EQ(diff_id, (res_index + 1) / 2);
        ASSERT_EQ(score, (dim * (diff_id * diff_id)));
    };
    runTopKSearchTest(parallel_index, query, k, verify_res);
    VecSimIndex_Free(parallel_index);
}

TYPED_TEST(HNSWTestParallel, parallelInsertMulti) {
    size_t n = 10000;
    size_t n_labels = 1000;
    size_t per_label = n / n_labels;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 16,
                         .efConstruction = 200};

    VecSimIndex *parallel_index = this->CreateNewIndex(params, true);
    size_t n_threads = 10;

    // Save the number fo tasks done by thread i in the i-th entry.
    std::vector<size_t> completed_tasks(n_threads, 0);
    auto parallel_insert = [&](int myID) {
        for (size_t i = myID; i < n; i += n_threads) {
            completed_tasks[myID]++;
            GenerateAndAddVector<TEST_DATA_T>(parallel_index, dim, i % n_labels, i);
        }
    };
    std::thread thread_objs[n_threads];
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i] = std::thread(parallel_insert, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
        thread_objs[i].join();
    }
    ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
    // Validate that every thread executed a single job.
    ASSERT_EQ(*std::min_element(completed_tasks.begin(), completed_tasks.end()), n / n_threads);
    ASSERT_EQ(*std::max_element(completed_tasks.begin(), completed_tasks.end()),
              ceil((double)n / n_threads));

    TEST_DATA_T query[dim];
    TEST_DATA_T query_val = (TEST_DATA_T)n / 2 + 10;
    GenerateVector<TEST_DATA_T>(query, dim, (TEST_DATA_T)query_val);
    auto verify_res = [&](size_t id, double score, size_t res_index) {
        // We expect to get the results with increasing order of the distance between the res
        // label and query_val%n_labels (that is ids 10, 9, 11, ... for the current arguments).
        // The score is the L2 distance between the vectors that correspond the ids.
        size_t diff_id = std::abs(int(id - (size_t)query_val % n_labels));
        ASSERT_EQ(diff_id, (res_index + 1) / 2);
        ASSERT_EQ(score, (dim * (diff_id * diff_id)));
    };
    runTopKSearchTest(parallel_index, query, k, verify_res);
    VecSimIndex_Free(parallel_index);
}

TYPED_TEST(HNSWTestParallel, parallelInsertSearch) {
    size_t n = 10000;
    size_t k = 11;
    size_t dim = 32;

    HNSWParams params = {.dim = dim,
                         .metric = VecSimMetric_L2,
                         .initialCapacity = n,
                         .M = 64,
                         .efConstruction = 200,
                         .efRuntime = n};

    for (bool is_multi : {true, false}) {
        VecSimIndex *parallel_index = this->CreateNewIndex(params, is_multi);
        size_t n_threads = MIN(10, std::thread::hardware_concurrency());
        // Save the number fo tasks done by thread i in the i-th entry.
        std::vector<size_t> completed_tasks(n_threads, 0);

        auto parallel_insert = [&](int myID) {
            for (labelType label = myID; label < n; label += n_threads / 2) {
                completed_tasks[myID]++;
                GenerateAndAddVector<TEST_DATA_T>(parallel_index, dim, label, label);
            }
        };

        TEST_DATA_T query_val = (TEST_DATA_T)n / 4;
        std::atomic_int successful_searches(0);
        auto parallel_knn_search = [&](int myID) {
            completed_tasks[myID]++;
            // Make sure were still indexing in parallel to the search (at most 90% if the vectors
            // were already indexed).
            ASSERT_LT(VecSimIndex_IndexSize(parallel_index), 0.9 * n);
            TEST_DATA_T query[dim];
            GenerateVector<TEST_DATA_T>(query, dim, query_val);
            auto verify_res = [&](size_t id, double score, size_t res_index) {
                // We expect to get the results with increasing order of the distance between the
                // res label and the query val (n/4, n/4-1, n/4+1, n/4-2, n/4+2, ...) The score is
                // the L2 distance between the vectors that correspond the ids.
                size_t diff_id = std::abs(int(id - query_val));
                ASSERT_EQ(diff_id, (res_index + 1) / 2);
                ASSERT_EQ(score, (dim * (diff_id * diff_id)));
            };
            runTopKSearchTest(parallel_index, query, k, verify_res);
            successful_searches++;
        };

        auto hnsw_index = this->CastToHNSW(parallel_index);
        std::thread thread_objs[n_threads];
        for (size_t i = 0; i < n_threads; i++) {
            if (i < n_threads / 2) {
                thread_objs[i] = std::thread(parallel_insert, i);
            } else {
                // Search threads are waiting in bust wait until the vectors of the query results
                // are done being indexed.
                bool wait_for_results = true;
                while (wait_for_results) {
                    wait_for_results = false;
                    for (labelType res_label = query_val - k / 2; res_label <= query_val + k / 2;
                         res_label++) {
                        if (!hnsw_index->safeCheckIfLabelExistsInIndex(res_label, true)) {
                            wait_for_results = true;
                            break; // results are not ready yet, restart the check.
                        }
                    }
                }
                thread_objs[i] = std::thread(parallel_knn_search, i);
            }
        }
        for (size_t i = 0; i < n_threads; i++) {
            thread_objs[i].join();
        }
        ASSERT_EQ(VecSimIndex_IndexSize(parallel_index), n);
        ASSERT_EQ(successful_searches, ceil(double(n_threads) / 2));
        // Validate that every thread executed a single job.
        ASSERT_EQ(
            *std::min_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
            n / (n_threads / 2));
        ASSERT_EQ(
            *std::max_element(completed_tasks.begin(), completed_tasks.begin() + n_threads / 2),
            ceil((double)n / (n_threads / 2)));
        ASSERT_EQ(*std::min_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
                  1);
        ASSERT_EQ(*std::max_element(completed_tasks.begin() + n_threads / 2, completed_tasks.end()),
                  1);
        VecSimIndex_Free(parallel_index);
    }
}
