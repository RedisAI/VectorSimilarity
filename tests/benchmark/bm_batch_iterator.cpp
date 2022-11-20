/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <benchmark/benchmark.h>
#include <random>
#include <unistd.h>
#include "VecSim/vec_sim.h"
#include "VecSim/query_results.h"
#include "bm_utils.h"

// Global benchmark data
size_t BM_VecSimBasics::n_vectors = 1000000;
size_t BM_VecSimBasics::n_queries = 10000;
size_t BM_VecSimBasics::dim = 768;
VecSimIndex *BM_VecSimBasics::bf_index;
VecSimIndex *BM_VecSimBasics::hnsw_index;
VecSimIndex *BM_VecSimBasics::ngt_index;
std::vector<std::vector<float>> BM_VecSimBasics::queries;
size_t BM_VecSimBasics::M = 64;
size_t BM_VecSimBasics::EF_C = 512;
size_t BM_VecSimBasics::block_size = 1024;
const char *BM_VecSimBasics::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
const char *BM_VecSimBasics::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimBasics::ref_count = 0;

class BM_BatchIterator : public BM_VecSimBasics {
protected:
    BM_BatchIterator() : BM_VecSimBasics() {}

    static void RunBatchedSearch_HNSW(benchmark::State &st, size_t &correct, size_t iter,
                                      size_t num_batches, size_t batch_size, size_t &total_res_num,
                                      size_t batch_increase_factor, size_t index_memory,
                                      double &memory_delta) {

        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(hnsw_index, queries[iter % n_queries].data(), nullptr);
        VecSimQueryResult_List accumulated_results[num_batches];
        size_t batch_num = 0;
        total_res_num = 0;

        // Run search in batches, collect the accumulated results.
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            if (batch_num == num_batches) {
                break;
            }
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, batch_size, BY_ID);
            total_res_num += VecSimQueryResult_Len(res);
            accumulated_results[batch_num++] = res;
            batch_size *= batch_increase_factor;
        }
        // Update the memory delta as a result of using the batch iterator.
        size_t curr_memory = VecSimIndex_Info(hnsw_index).hnswInfo.memory;
        memory_delta += (double)(curr_memory - index_memory);
        VecSimBatchIterator_Free(batchIterator);
        st.PauseTiming();

        // Measure recall - compare every result that was collected in some batch to the BF results.
        auto bf_results = VecSimIndex_TopKQuery(bf_index, queries[iter % n_queries].data(),
                                                total_res_num, nullptr, BY_SCORE);
        for (size_t i = 0; i < batch_num; i++) {
            auto hnsw_results = accumulated_results[i];
            auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
            while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
                auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
                auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
                while (VecSimQueryResult_IteratorHasNext(bf_it)) {
                    auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
                    if (VecSimQueryResult_GetId(hnsw_res_item) ==
                        VecSimQueryResult_GetId(bf_res_item)) {
                        correct++;
                        break;
                    }
                }
                VecSimQueryResult_IteratorFree(bf_it);
            }
            VecSimQueryResult_IteratorFree(hnsw_it);
            VecSimQueryResult_Free(hnsw_results);
        }
        VecSimQueryResult_Free(bf_results);
        st.ResumeTiming();
    }

    ~BM_BatchIterator() = default;
};

BENCHMARK_DEFINE_F(BM_BatchIterator, BF_FixedBatchSize)(benchmark::State &st) {

    size_t batch_size = st.range(0);
    size_t num_batches = st.range(1);
    size_t iter = 0;
    size_t index_memory = VecSimIndex_Info(bf_index).bfInfo.memory;
    double memory_delta = 0.0;
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(bf_index, queries[iter % n_queries].data(), nullptr);
        size_t batches_counter = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, batch_size, BY_ID);
            VecSimQueryResult_Free(res);
            batches_counter++;
            if (batches_counter == num_batches) {
                break;
            }
        }
        size_t curr_memory = VecSimIndex_Info(bf_index).bfInfo.memory;
        memory_delta += (double)(curr_memory - index_memory);
        VecSimBatchIterator_Free(batchIterator);
        iter++;
    }
    st.counters["memory"] = memory_delta / (double)iter;
}

BENCHMARK_DEFINE_F(BM_BatchIterator, BF_VariableBatchSize)(benchmark::State &st) {

    size_t batch_size = st.range(0);
    size_t num_batches = st.range(1);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(bf_index, queries[iter % n_queries].data(), nullptr);
        size_t batches_counter = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, batch_size, BY_ID);
            VecSimQueryResult_Free(res);
            batches_counter++;
            batch_size *= 2;
            if (batches_counter == num_batches) {
                break;
            }
        }
        VecSimBatchIterator_Free(batchIterator);
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_BatchIterator, BF_BatchesToAdhocBF)(benchmark::State &st) {

    size_t step = st.range(0);
    size_t num_batches = st.range(1);
    size_t batch_size = 10;
    size_t iter = 0;
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(bf_index, queries[iter % n_queries].data(), nullptr);
        size_t batches_counter = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            if (batches_counter == num_batches) {
                break;
            }
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, batch_size, BY_ID);
            VecSimQueryResult_Free(res);
            batches_counter++;
            batch_size *= 2;
        }
        VecSimBatchIterator_Free(batchIterator);
        // Switch to ad-hoc BF
        for (size_t i = 0; i < n_vectors; i += step) {
            VecSimIndex_GetDistanceFrom(bf_index, i, queries[iter % n_queries].data());
        }
        iter++;
    }
}

BENCHMARK_DEFINE_F(BM_BatchIterator, HNSW_FixedBatchSize)(benchmark::State &st) {

    size_t batch_size = st.range(0);
    size_t num_batches = st.range(1);
    size_t total_res_num = num_batches * batch_size;
    size_t iter = 0;
    size_t correct = 0;
    size_t index_memory = VecSimIndex_Info(hnsw_index).hnswInfo.memory;
    double memory_delta = 0.0;

    for (auto _ : st) {
        RunBatchedSearch_HNSW(st, correct, iter, num_batches, batch_size, total_res_num, 1,
                              index_memory, memory_delta);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (total_res_num * iter);
    st.counters["memory"] = memory_delta / (double)iter;
}

BENCHMARK_DEFINE_F(BM_BatchIterator, HNSW_VariableBatchSize)(benchmark::State &st) {

    size_t initial_batch_size = st.range(0);
    size_t num_batches = st.range(1);
    size_t total_res_num;
    size_t iter = 0;
    size_t correct = 0;
    size_t index_memory = VecSimIndex_Info(hnsw_index).hnswInfo.memory;
    double memory_delta = 0.0;

    for (auto _ : st) {
        RunBatchedSearch_HNSW(st, correct, iter, num_batches, initial_batch_size, total_res_num, 2,
                              index_memory, memory_delta);
        iter++;
    }
    st.counters["Recall"] = (float)correct / (float)(total_res_num * iter);
    st.counters["memory"] = memory_delta / (double)iter;
}

BENCHMARK_DEFINE_F(BM_BatchIterator, HNSW_BatchesToAdhocBF)(benchmark::State &st) {

    size_t step = st.range(0);
    size_t num_batches = st.range(1);
    size_t total_res_num;
    size_t iter = 0;
    size_t correct = 0;
    size_t index_memory = VecSimIndex_Info(hnsw_index).hnswInfo.memory;
    double memory_delta = 0.0;

    for (auto _ : st) {
        RunBatchedSearch_HNSW(st, correct, iter, num_batches, 10, total_res_num, 2, index_memory,
                              memory_delta);
        // Switch to ad-hoc BF
        for (size_t i = 0; i < n_vectors; i += step) {
            VecSimIndex_GetDistanceFrom(hnsw_index, i, queries[iter % n_queries].data());
        }
        iter++;
    }
    st.counters["memory"] = memory_delta / (double)iter;
}

// Register the functions as a benchmark
BENCHMARK_REGISTER_F(BM_BatchIterator, BF_FixedBatchSize)
// {batch_size, num_batches}
FIXED_BATCH_SIZE_ARGS(10, 1)
FIXED_BATCH_SIZE_ARGS(10, 3)
FIXED_BATCH_SIZE_ARGS(10, 5)
FIXED_BATCH_SIZE_ARGS(100, 1)
FIXED_BATCH_SIZE_ARGS(100, 3)
FIXED_BATCH_SIZE_ARGS(100, 5)
FIXED_BATCH_SIZE_ARGS(1000, 1)
FIXED_BATCH_SIZE_ARGS(1000, 3)
FIXED_BATCH_SIZE_ARGS(1000, 5)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_BatchIterator, BF_VariableBatchSize)
// {initial_batch_size, num_batches}
// batch size is increased by factor of 2 from iteration to the next one.
VARIABLE_BATCH_SIZE_ARGS(10, 2)
VARIABLE_BATCH_SIZE_ARGS(10, 4)
VARIABLE_BATCH_SIZE_ARGS(100, 2)
VARIABLE_BATCH_SIZE_ARGS(100, 4)
VARIABLE_BATCH_SIZE_ARGS(1000, 2)
VARIABLE_BATCH_SIZE_ARGS(1000, 4)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_BatchIterator, BF_BatchesToAdhocBF)
// {step, num_batches} - where step is the ratio between the index size to the number of vectors to
// go over in ad-hoc BF.
// batch size is increased by factor of 2 from iteration to the next one, and initial batch size
// is 10.
BATCHES_TO_ADHOC_ARGS(5, 0)
BATCHES_TO_ADHOC_ARGS(5, 2)
BATCHES_TO_ADHOC_ARGS(5, 5)
BATCHES_TO_ADHOC_ARGS(10, 0)
BATCHES_TO_ADHOC_ARGS(10, 2)
BATCHES_TO_ADHOC_ARGS(10, 5)
BATCHES_TO_ADHOC_ARGS(20, 0)
BATCHES_TO_ADHOC_ARGS(20, 2)
BATCHES_TO_ADHOC_ARGS(20, 5)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_BatchIterator, HNSW_FixedBatchSize)
// {batch_size, num_batches}
FIXED_BATCH_SIZE_ARGS(10, 1)
FIXED_BATCH_SIZE_ARGS(10, 3)
FIXED_BATCH_SIZE_ARGS(10, 5)
FIXED_BATCH_SIZE_ARGS(100, 1)
FIXED_BATCH_SIZE_ARGS(100, 3)
FIXED_BATCH_SIZE_ARGS(100, 5)
FIXED_BATCH_SIZE_ARGS(1000, 1)
FIXED_BATCH_SIZE_ARGS(1000, 3)
FIXED_BATCH_SIZE_ARGS(1000, 5)->Iterations(50)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_BatchIterator, HNSW_VariableBatchSize)
// {initial_batch_size, num_batches}
// batch size is increased by factor of 2 from iteration to the next one.
VARIABLE_BATCH_SIZE_ARGS(10, 2)
VARIABLE_BATCH_SIZE_ARGS(10, 4)
VARIABLE_BATCH_SIZE_ARGS(100, 2)
VARIABLE_BATCH_SIZE_ARGS(100, 4)
VARIABLE_BATCH_SIZE_ARGS(1000, 2)
VARIABLE_BATCH_SIZE_ARGS(1000, 4)->Iterations(50)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(BM_BatchIterator, HNSW_BatchesToAdhocBF)
// {step, num_batches} - where step is the ratio between the index size to the number of vectors to
// go over in ad-hoc BF.
// batch size is increased by factor of 2 from iteration to the next one, and initial batch size
// is 10.
BATCHES_TO_ADHOC_ARGS(5, 0)
BATCHES_TO_ADHOC_ARGS(5, 2)
BATCHES_TO_ADHOC_ARGS(5, 5)
BATCHES_TO_ADHOC_ARGS(10, 0)
BATCHES_TO_ADHOC_ARGS(10, 2)
BATCHES_TO_ADHOC_ARGS(10, 5)
BATCHES_TO_ADHOC_ARGS(20, 0)
BATCHES_TO_ADHOC_ARGS(20, 2)
BATCHES_TO_ADHOC_ARGS(20, 5)
BATCHES_TO_ADHOC_ARGS(50, 0)
BATCHES_TO_ADHOC_ARGS(50, 2)
BATCHES_TO_ADHOC_ARGS(50, 5)
BATCHES_TO_ADHOC_ARGS(100, 0)->Iterations(50)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
