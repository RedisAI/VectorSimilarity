#pragma once
#include <atomic>
#include "bm_common.h"
#include <chrono>

using namespace std::chrono;

template <typename index_type_t>
class BM_VecSimBasics : public BM_VecSimCommon<index_type_t> {
public:
    using data_t = typename index_type_t::data_t;

    BM_VecSimBasics() = default;
    ~BM_VecSimBasics() = default;

    // Add one label in each iteration. For multi index adds multiple vectors under the same label
    // per iteration, for single index adds one vector per iteration.
    static void AddLabel(benchmark::State &st);

    static void AddLabel_AsyncIngest(benchmark::State &st);

    static void DeleteLabel_AsyncRepair(benchmark::State &st);

    // We pass a specific index pointer instead of VecSimIndex * so we can use GetDataByLabel
    // which is not known to VecSimIndex class.
    // We delete one label in each iteration. For multi index deletes multiple vectors per
    // iteration, for single index deletes one vector per iteration.
    template <typename algo_t>
    static void DeleteLabel(algo_t *index, benchmark::State &st);

    static void Range_BF(benchmark::State &st);
    static void Range_HNSW(benchmark::State &st);

private:
    // Vectors of vector to store deleted labels' data.
    using LabelData = std::vector<std::vector<data_t>>;
};

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::AddLabel(benchmark::State &st) {

    auto index = INDICES[st.range(0)];
    size_t index_size = N_VECTORS;
    size_t initial_label_count = index->indexLabelCount();

    // In a single vector per label index, index size should equal label count.
    size_t vec_per_label = index_size % initial_label_count == 0
                               ? index_size / initial_label_count
                               : index_size / initial_label_count + 1;
    labelType label = initial_label_count;
    size_t added_vec_count = 0;

    size_t memory_delta = index->getAllocationSize();
    // Add a new label from the test set in every iteration.
    for (auto _ : st) {
        // Add one label
        for (labelType vec = 0; vec < vec_per_label; ++vec) {
            VecSimIndex_AddVector(index, QUERIES[added_vec_count % N_QUERIES].data(), label);
        }
        added_vec_count += vec_per_label;
        label++;
    }
    memory_delta = index->getAllocationSize() - memory_delta;
    // For tiered index, wait for all threads to finish indexing
    BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();

    st.counters["memory_per_vector"] = (double)memory_delta / (double)added_vec_count;
    st.counters["vectors_per_label"] = vec_per_label;

    assert(VecSimIndex_IndexSize(index) == N_VECTORS + added_vec_count);

    // Clean-up all the new vectors to restore the index size to its original value.
    // Note we loop over the new labels and not the internal ids. This way in multi indices BM all
    // the new vectors added under the same label will be removed in one call.
    size_t new_label_count = index->indexLabelCount();
    for (size_t label = initial_label_count; label < new_label_count; label++) {
        // If index is tiered HNSW, remove directly from the underline HNSW.
        VecSimIndex_DeleteVector(
            INDICES[st.range(0) == VecSimAlgo_TIERED ? VecSimAlgo_HNSWLIB : st.range(0)], label);
    }
    assert(VecSimIndex_IndexSize(index) == N_VECTORS);
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::AddLabel_AsyncIngest(benchmark::State &st) {

    size_t index_size = N_VECTORS;
    size_t initial_label_count = (INDICES[st.range(0)])->indexLabelCount();

    // In a single vector per label index, index size should equal label count.
    size_t vec_per_label = index_size % initial_label_count == 0
                               ? index_size / initial_label_count
                               : index_size / initial_label_count + 1;
    size_t memory_before = (INDICES[st.range(0)])->getAllocationSize();
    labelType label = initial_label_count;
    size_t added_vec_count = 0;

    // Add a new label from the test set in every iteration.
    for (auto _ : st) {
        // Add one label
        for (labelType vec = 0; vec < vec_per_label; ++vec) {
            VecSimIndex_AddVector(INDICES[st.range(0)], QUERIES[added_vec_count % N_QUERIES].data(),
                                  label);
        }
        added_vec_count += vec_per_label;
        label++;
        if (label == initial_label_count + BM_VecSimGeneral::block_size) {
            BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();
        }
    }

    size_t memory_delta = (INDICES[st.range(0)])->getAllocationSize() - memory_before;
    st.counters["memory_per_vector"] = (double)memory_delta / (double)added_vec_count;
    st.counters["vectors_per_label"] = vec_per_label;
    st.counters["num_threads"] = BM_VecSimGeneral::mock_thread_pool.thread_pool_size;

    size_t index_size_after = VecSimIndex_IndexSize(INDICES[st.range(0)]);
    assert(index_size_after == N_VECTORS + added_vec_count);

    // Clean-up all the new vectors to restore the index size to its original value.
    // Note we loop over the new labels and not the internal ids. This way in multi indices BM all
    // the new vectors added under the same label will be removed in one call.
    size_t new_label_count = (INDICES[st.range(0)])->indexLabelCount();
    // Remove directly inplace from the underline HNSW index.
    for (size_t label_ = initial_label_count; label_ < new_label_count; label_++) {
        VecSimIndex_DeleteVector(INDICES[VecSimAlgo_HNSWLIB], label_);
    }

    assert(VecSimIndex_IndexSize(INDICES[st.range(0)]) == N_VECTORS);
}

template <typename index_type_t>
template <typename algo_t>
void BM_VecSimBasics<index_type_t>::DeleteLabel(algo_t *index, benchmark::State &st) {
    // Remove a different vector in every execution.
    size_t label_to_remove = 0;
    double memory_delta, memory_before = index->getAllocationSize();
    size_t removed_vectors_count = 0;
    std::vector<LabelData> removed_labels_data;

    for (auto _ : st) {
        st.PauseTiming();
        LabelData data(0);
        // Get label id(s) data.
        index->getDataByLabel(label_to_remove, data);

        removed_labels_data.push_back(data);

        removed_vectors_count += data.size();
        st.ResumeTiming();

        // Delete label
        VecSimIndex_DeleteVector(index, label_to_remove++);
    }
    memory_delta = index->getAllocationSize() - memory_before;

    BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();
    // Remove the rest of the vectors that hadn't been swapped yet for tiered index.
    if (VecSimIndex_BasicInfo(index).algo == VecSimAlgo_TIERED) {
        reinterpret_cast<TieredHNSWIndex<data_t, data_t> *>(index)->executeReadySwapJobs();
    }
    st.counters["memory_per_vector"] = memory_delta / (double)removed_vectors_count;

    // Restore index state.
    // For each label in removed_labels_data
    for (size_t label_idx = 0; label_idx < removed_labels_data.size(); label_idx++) {
        size_t vec_count = removed_labels_data[label_idx].size();
        // Reinsert all the deleted vectors under this label.
        for (size_t vec_idx = 0; vec_idx < vec_count; ++vec_idx) {
            VecSimIndex_AddVector(index, removed_labels_data[label_idx][vec_idx].data(), label_idx);
        }
    }
    BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();
    size_t cur_index_size = VecSimIndex_IndexSize(index);
    assert(VecSimIndex_IndexSize(index) == N_VECTORS);
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::DeleteLabel_AsyncRepair(benchmark::State &st) {
    // Remove a different vector in every execution.
    size_t label_to_remove = 0;
    auto *tiered_index =
        reinterpret_cast<TieredHNSWIndex<data_t, data_t> *>(INDICES[VecSimAlgo_TIERED]);
    int memory_before = tiered_index->getAllocationSize();
    size_t removed_vectors_count = 0;
    std::vector<LabelData> removed_labels_data;
    tiered_index->pendingSwapJobsThreshold = st.range(0);

    for (auto _ : st) {
        st.PauseTiming();
        LabelData data(0);
        // Get label id(s) data.
        tiered_index->getDataByLabel(label_to_remove, data);

        removed_labels_data.push_back(data);

        removed_vectors_count += data.size();
        st.ResumeTiming();

        // Delete label
        VecSimIndex_DeleteVector(tiered_index, label_to_remove++);
        if (label_to_remove == BM_VecSimGeneral::block_size) {
            BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();
        }
    }

    // Avg. memory delta per vector equals the total memory delta divided by the number
    // of deleted vectors.
    int memory_delta = tiered_index->getAllocationSize() - memory_before;
    st.counters["memory_per_vector"] = memory_delta / (double)removed_vectors_count;
    st.counters["num_threads"] = (double)BM_VecSimGeneral::mock_thread_pool.thread_pool_size;
    st.counters["num_zombies"] = tiered_index->idToSwapJob.size();

    // Remove the rest of the vectors that hadn't been swapped yet.
    auto start = high_resolution_clock::now();
    tiered_index->pendingSwapJobsThreshold = 1;
    tiered_index->executeReadySwapJobs();
    tiered_index->pendingSwapJobsThreshold = DEFAULT_PENDING_SWAP_JOBS_THRESHOLD;
    auto end = high_resolution_clock::now();
    st.counters["cleanup_time"] = (double)duration_cast<milliseconds>(end - start).count();

    // Restore index state.
    // For each label in removed_labels_data
    for (size_t label_idx = 0; label_idx < removed_labels_data.size(); label_idx++) {
        size_t vec_count = removed_labels_data[label_idx].size();
        // Reinsert all the deleted vectors under this label.
        for (size_t vec_idx = 0; vec_idx < vec_count; ++vec_idx) {
            VecSimIndex_AddVector(tiered_index, removed_labels_data[label_idx][vec_idx].data(),
                                  label_idx);
        }
    }
    BM_VecSimGeneral::mock_thread_pool.thread_pool_wait();
    assert(VecSimIndex_IndexSize(tiered_index) == N_VECTORS);
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Range_BF(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    size_t iter = 0;
    size_t total_res = 0;

    for (auto _ : st) {
        auto res = VecSimIndex_RangeQuery(INDICES[VecSimAlgo_BF], QUERIES[iter % N_QUERIES].data(),
                                          radius, nullptr, BY_ID);
        total_res += VecSimQueryResult_Len(res);
        iter++;
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
}

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::Range_HNSW(benchmark::State &st) {
    double radius = (1.0 / 100.0) * (double)st.range(0);
    double epsilon = (1.0 / 1000.0) * (double)st.range(1);
    size_t iter = 0;
    size_t total_res = 0;
    size_t total_res_bf = 0;
    HNSWRuntimeParams hnswRuntimeParams = {.epsilon = epsilon};
    auto query_params = BM_VecSimGeneral::CreateQueryParams(hnswRuntimeParams);

    for (auto _ : st) {
        auto hnsw_results =
            VecSimIndex_RangeQuery(INDICES[VecSimAlgo_HNSWLIB], QUERIES[iter % N_QUERIES].data(),
                                   radius, &query_params, BY_ID);
        st.PauseTiming();
        total_res += VecSimQueryResult_Len(hnsw_results);

        // Measure recall:
        auto bf_results = VecSimIndex_RangeQuery(
            INDICES[VecSimAlgo_BF], QUERIES[iter % N_QUERIES].data(), radius, nullptr, BY_ID);
        total_res_bf += VecSimQueryResult_Len(bf_results);

        VecSimQueryResult_Free(bf_results);
        VecSimQueryResult_Free(hnsw_results);
        iter++;
        st.ResumeTiming();
    }
    st.counters["Avg. results number"] = (double)total_res / iter;
    st.counters["Recall"] = (float)total_res / total_res_bf;
}

#define UNIT_AND_ITERATIONS Unit(benchmark::kMillisecond)->Iterations(BM_VecSimGeneral::block_size)

// The actual radius will be the given arg divided by 100, since arg must be an integer.
#define REGISTER_Range_BF(BM_FUNC)                                                                 \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Arg(20)                                                                                  \
        ->Arg(35)                                                                                  \
        ->Arg(50)                                                                                  \
        ->ArgName("radiusX100")                                                                    \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

// {radius*100, epsilon*1000}
// The actual radius will be the given arg divided by 100, and the actual epsilon values
// will be the given arg divided by 1000.
#define REGISTER_Range_HNSW(BM_FUNC)                                                               \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->Args({20, 1})                                                                            \
        ->Args({20, 10})                                                                           \
        ->Args({20, 100})                                                                          \
        ->Args({35, 1})                                                                            \
        ->Args({35, 10})                                                                           \
        ->Args({35, 100})                                                                          \
        ->Args({50, 1})                                                                            \
        ->Args({50, 10})                                                                           \
        ->Args({50, 100})                                                                          \
        ->ArgNames({"radiusX100", "epsilonX1000"})                                                 \
        ->Iterations(10)                                                                           \
        ->Unit(benchmark::kMillisecond)

#define REGISTER_AddLabel(BM_FUNC, VecSimAlgo)                                                     \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo)                                                     \
        ->ArgName(#VecSimAlgo)

// DeleteLabel define and register macros
#define DEFINE_DELETE_LABEL(BM_FUNC, INDEX_TYPE, INDEX_NAME, DATA_TYPE, DIST_TYPE, VecSimAlgo)     \
    BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics, BM_FUNC, INDEX_TYPE)(benchmark::State & st) {     \
        DeleteLabel<INDEX_NAME<DATA_TYPE, DIST_TYPE>>(                                             \
            reinterpret_cast<INDEX_NAME<DATA_TYPE, DIST_TYPE> *>(                                  \
                BM_VecSimIndex<INDEX_TYPE>::indices[VecSimAlgo]),                                  \
            st);                                                                                   \
    }
#define REGISTER_DeleteLabel(BM_FUNC)                                                              \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)->UNIT_AND_ITERATIONS
