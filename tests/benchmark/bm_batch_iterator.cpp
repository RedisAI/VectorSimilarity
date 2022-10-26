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
#include "bm_common.h"

// Global benchmark data
size_t BM_VecSimUtils::n_vectors = 1000000;
size_t BM_VecSimUtils::n_queries = 10000;
size_t BM_VecSimUtils::dim = 768;
size_t BM_VecSimUtils::block_size = 1024;
const char *BM_VecSimUtils::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
const char *BM_VecSimUtils::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";

size_t BM_VecSimUtils::ref_count = 0;

template <typename index_type_t>
class BM_BatchIterator : public BM_VecSimIndex<index_type_t> {
protected:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    BM_BatchIterator() : BM_VecSimIndex<index_type_t>(false){};
    ~BM_BatchIterator() = default;

    static void BF_FixedBatchSize(size_t batch_size, size_t num_batches, benchmark::State &st);

    static void RunBatchedSearch_HNSW(benchmark::State &st, size_t &correct, size_t iter,
                                      size_t num_batches, size_t batch_size, size_t &total_res_num,
                                      size_t batch_increase_factor, size_t index_memory,
                                      double &memory_delta);
};

template <typename index_type_t>
void BM_BatchIterator<index_type_t>::RunBatchedSearch_HNSW(benchmark::State &st, size_t &correct,
                                                           size_t iter, size_t num_batches,
                                                           size_t batch_size, size_t &total_res_num,
                                                           size_t batch_increase_factor,
                                                           size_t index_memory,
                                                           double &memory_delta) {
    VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(
        INDICES[VecSimAlgo_HNSWLIB], QUERIES[iter % N_QUERIES].data(), nullptr);
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
    size_t curr_memory = VecSimIndex_Info(INDICES[VecSimAlgo_HNSWLIB]).hnswInfo.memory;
    memory_delta += (double)(curr_memory - index_memory);
    VecSimBatchIterator_Free(batchIterator);
    st.PauseTiming();

    // Measure recall - compare every result that was collected in some batch to the BF results.
    auto bf_results = VecSimIndex_TopKQuery(
        INDICES[VecSimAlgo_BF], QUERIES[iter % N_QUERIES].data(), total_res_num, nullptr, BY_SCORE);
    for (size_t i = 0; i < batch_num; i++) {
        auto hnsw_results = accumulated_results[i];
        BM_VecSimGeneral::MeasureRecall(hnsw_results, bf_results, correct);
        VecSimQueryResult_Free(hnsw_results);
    }
    VecSimQueryResult_Free(bf_results);
    st.ResumeTiming();
}

template <typename index_type_t>
void BM_BatchIterator<index_type_t>::BF_FixedBatchSize(size_t batch_size, size_t num_batches,
                                                       benchmark::State &st) {
    size_t iter = 0;
    size_t index_memory = VecSimIndex_Info(INDICES[VecSimAlgo_BF]).bfInfo.memory;
    double memory_delta = 0.0;
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator = VecSimBatchIterator_New(
            INDICES[VecSimAlgo_BF], QUERIES[iter % N_QUERIES].data(), nullptr);
        size_t batches_counter = 0;
        while (VecSimBatchIterator_HasNext(batchIterator)) {
            VecSimQueryResult_List res = VecSimBatchIterator_Next(batchIterator, batch_size, BY_ID);
            VecSimQueryResult_Free(res);
            batches_counter++;
            if (batches_counter == num_batches) {
                break;
            }
        }
        size_t curr_memory = VecSimIndex_Info(INDICES[VecSimAlgo_BF]).bfInfo.memory;
        memory_delta += (double)(curr_memory - index_memory);
        VecSimBatchIterator_Free(batchIterator);
        iter++;
    }
    st.counters["memory"] = memory_delta / (double)iter;
}
BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, BF_FixedBatchSize_fp32, fp32_index_t)
(benchmark::State &st) {
    size_t batch_size = st.range(0);
    size_t num_batches = st.range(1);

    BF_FixedBatchSize(batch_size, num_batches, st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_BatchIterator, BF_FixedBatchSize_fp64, fp64_index_t)
(benchmark::State &st) {
    size_t batch_size = st.range(0);
    size_t num_batches = st.range(1);

    BF_FixedBatchSize(batch_size, num_batches, st);
}

#define FIXED_BATCH_SIZE_ARGS(batch_size, num_batches)                                             \
    Args({batch_size, num_batches})->ArgNames({"batch size", "number of batches"})

#define REGISTER_BF_FixedBatchSize(BM_FUNC)                                                        \
    BENCHMARK_REGISTER_F(BM_BatchIterator, BM_FUNC)                                                \
        ->FIXED_BATCH_SIZE_ARGS(10, 1)                                                             \
        ->FIXED_BATCH_SIZE_ARGS(10, 3)                                                             \
        ->FIXED_BATCH_SIZE_ARGS(10, 5)                                                             \
        ->FIXED_BATCH_SIZE_ARGS(100, 1)                                                            \
        ->FIXED_BATCH_SIZE_ARGS(100, 3)                                                            \
        ->FIXED_BATCH_SIZE_ARGS(100, 5)                                                            \
        ->FIXED_BATCH_SIZE_ARGS(1000, 1)                                                           \
        ->FIXED_BATCH_SIZE_ARGS(1000, 3)                                                           \
        ->FIXED_BATCH_SIZE_ARGS(1000, 5)                                                           \
        ->Unit(benchmark::kMillisecond);

// Register the functions as a benchmark
// {batch_size, num_batches}
REGISTER_BF_FixedBatchSize(BF_FixedBatchSize_fp32);
REGISTER_BF_FixedBatchSize(BF_FixedBatchSize_fp64);

BENCHMARK_DEFINE_F(BM_BatchIterator, BF_VariableBatchSize)(benchmark::State &st) {

    size_t batch_size = st.range(0);
    size_t num_batches = st.range(1);
    size_t iter = 0;
    for (auto _ : st) {
        VecSimBatchIterator *batchIterator =
            VecSimBatchIterator_New(bf_index, (*queries)[iter % n_queries].data(), nullptr);
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
            VecSimBatchIterator_New(bf_index, (*queries)[iter % n_queries].data(), nullptr);
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
            VecSimIndex_GetDistanceFrom(bf_index, i, (*queries)[iter % n_queries].data());
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
            VecSimIndex_GetDistanceFrom(hnsw_index, i, (*queries)[iter % n_queries].data());
        }
        iter++;
    }
    st.counters["memory"] = memory_delta / (double)iter;
}

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
