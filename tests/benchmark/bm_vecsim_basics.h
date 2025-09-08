#pragma once
#include "bm_common.h"

template <typename index_type_t>
class BM_VecSimBasics : public BM_VecSimCommon<index_type_t> {
public:
    using data_t = typename index_type_t::data_t;

    BM_VecSimBasics() = default;
    ~BM_VecSimBasics() = default;

    // Add one label in each iteration. For multi index adds multiple vectors under the same label
    // per iteration, for single index adds one vector per iteration.
    static void AddLabel(benchmark::State &st);

    // We pass a specific index pointer instead of VecSimIndex * so we can use GetDataByLabel
    // which is not known to VecSimIndex class.
    // We delete one label in each iteration. For multi index deletes multiple vectors per
    // iteration, for single index deletes one vector per iteration.
    template <typename algo_t>
    static void DeleteLabel(algo_t *index, benchmark::State &st);

    static void Range_BF(benchmark::State &st);
    static void Range_HNSW(benchmark::State &st);

    // Reproduces allocation/deallocation oscillation issue at block size boundaries.
    // Sets up index at blockSize+1 capacity, then repeatedly deletes and re-adds the same vector,
    // triggering constant grow-shrink cycles.
    // This behavior was fixed by PR #753 with a conservative resize strategy that only
    // shrinks containers when there are 2+ free blocks, preventing oscillation cycles.
    // Expected: High allocation overhead before fix, stable performance after fix.
    static void UpdateAtBlockSize(benchmark::State &st);

private:
    // Vectors of vector to store deleted labels' data.
    using LabelData = std::vector<std::vector<data_t>>;
};

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::AddLabel(benchmark::State &st) {

    size_t index_size = N_VECTORS;
    size_t initial_label_count = (INDICES[st.range(0)])->indexLabelCount();

    // In a single vector per label index, index size should equal label count.
    size_t vec_per_label = index_size % initial_label_count == 0
                               ? index_size / initial_label_count
                               : index_size / initial_label_count + 1;
    size_t memory_delta = 0;
    labelType label = initial_label_count;
    size_t added_vec_count = 0;

    // Add a new label from the test set in every iteration.
    for (auto _ : st) {
        // Add one label
        for (labelType vec = 0; vec < vec_per_label; ++vec) {
            memory_delta += VecSimIndex_AddVector(
                INDICES[st.range(0)], QUERIES[added_vec_count % N_QUERIES].data(), label);
        }
        added_vec_count += vec_per_label;
        label++;
    }

    st.counters["memory_per_vector"] =
        benchmark::Counter((double)memory_delta / (double)added_vec_count,
                           benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);
    st.counters["vectors_per_label"] = vec_per_label;

    assert(VecSimIndex_IndexSize(INDICES[st.range(0)]) == N_VECTORS + added_vec_count);

    // Clean-up all the new vectors to restore the index size to its original value.
    // Note we loop over the new labels and not the internal ids. This way in multi indices BM all
    // the new vectors added under the same label will be removed in one call.
    size_t new_label_count = (INDICES[st.range(0)])->indexLabelCount();
    for (size_t label = initial_label_count; label < new_label_count; label++) {
        VecSimIndex_DeleteVector(INDICES[st.range(0)], label);
    }

    assert(VecSimIndex_IndexSize(INDICES[st.range(0)]) == N_VECTORS);
}
template <typename index_type_t>
template <typename algo_t>
void BM_VecSimBasics<index_type_t>::DeleteLabel(algo_t *index, benchmark::State &st) {
    // Remove a different vector in every execution.
    size_t label_to_remove = 0;
    double memory_delta = 0;
    size_t removed_vectors_count = 0;
    std::vector<LabelData> removed_labels_data;

    for (auto _ : st) {
        st.PauseTiming();
        LabelData data(0);
        // Get label id(s) data.
        index->GetDataByLabel(label_to_remove, data);

        removed_labels_data.push_back(data);

        removed_vectors_count += data.size();
        st.ResumeTiming();

        // Delete label
        auto delta = (double)VecSimIndex_DeleteVector(index, label_to_remove++);
        memory_delta += delta;
    }

    // Avg. memory delta per vector equals the total memory delta divided by the number
    // of deleted vectors.
    st.counters["memory_per_vector"] =
        benchmark::Counter((double)memory_delta / (double)removed_vectors_count,
                           benchmark::Counter::kDefaults, benchmark::Counter::OneK::kIs1024);

    // Restore index state.
    // For each label in removed_labels_data
    for (size_t label_idx = 0; label_idx < removed_labels_data.size(); label_idx++) {
        size_t vec_count = removed_labels_data[label_idx].size();
        // Reinsert all the deleted vectors under this label.
        for (size_t vec_idx = 0; vec_idx < vec_count; ++vec_idx) {
            VecSimIndex_AddVector(index, removed_labels_data[label_idx][vec_idx].data(), label_idx);
        }
    }
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

template <typename index_type_t>
void BM_VecSimBasics<index_type_t>::UpdateAtBlockSize(benchmark::State &st) {
    auto index = INDICES[st.range(0)];
    size_t initial_index_size = VecSimIndex_IndexSize(index);
    // Calculate vectors needed to reach next block boundary
    size_t vecs_to_blocksize =
        BM_VecSimGeneral::block_size - (initial_index_size % BM_VecSimGeneral::block_size);
    assert(vecs_to_blocksize < BM_VecSimGeneral::block_size);
    labelType initial_label_count = index->indexLabelCount();
    labelType curr_label = initial_label_count;
    // Set up index at blockSize+1 to trigger oscillation issue
    // Make sure we have enough queries to add a new label.
    assert(N_QUERIES > BM_VecSimGeneral::block_size);
    size_t overhead = 1;
    size_t added_vec_count = vecs_to_blocksize + overhead;
    for (size_t i = 0; i < added_vec_count; ++i) {
        VecSimIndex_AddVector(index, QUERIES[added_vec_count % N_QUERIES].data(), curr_label++);
    }
    assert(VecSimIndex_IndexSize(index) % BM_VecSimGeneral::block_size == overhead);
    assert(VecSimIndex_IndexSize(index) == N_VECTORS + added_vec_count);
    std::cout << "Added " << added_vec_count << " vectors to reach block size boundary."
              << std::endl;
    std::cout << "Index size is now " << VecSimIndex_IndexSize(index) << std::endl;
    std::cout << "Last label is " << curr_label - 1 << std::endl;
    // Benchmark loop: repeatedly delete/add same vector to trigger grow-shrink cycles
    labelType label_to_update = curr_label - 1;
    size_t index_cap = index->indexCapacity();
    for (auto _ : st) {
        // Remove the vector directly from hnsw
        size_t ret = VecSimIndex_DeleteVector(index, label_to_update);
        assert(ret == 1);
        assert(index->indexCapacity() == index_cap - BM_VecSimGeneral::block_size);
        // Capacity should shrink by one block after deletion
        ret = VecSimIndex_AddVector(index, QUERIES[(added_vec_count - 1) % N_QUERIES].data(),
                                    label_to_update);
        assert(ret == 1);
        assert(VecSimIndex_IndexSize(index) == N_VECTORS + added_vec_count);
        // Capacity should grow back to original size after addition
        assert(index->indexCapacity() == index_cap);
    }
    assert(VecSimIndex_IndexSize(index) == N_VECTORS + added_vec_count);
    // Clean-up all the new vectors to restore the index size to its original value.
    size_t new_label_count = index->indexLabelCount();
    for (size_t label = initial_label_count; label < new_label_count; label++) {
        // If index is tiered HNSW, remove directly from the underline HNSW.
        VecSimIndex_DeleteVector(index, label);
    }
    assert(VecSimIndex_IndexSize(index) == N_VECTORS);
}

#define UNIT_AND_ITERATIONS                                                                        \
    Unit(benchmark::kMillisecond)->Iterations((long)BM_VecSimGeneral::block_size)

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

#define REGISTER_UpdateAtBlockSize(BM_FUNC, VecSimAlgo)                                            \
    BENCHMARK_REGISTER_F(BM_VecSimBasics, BM_FUNC)                                                 \
        ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo)                                                     \
        ->ArgName(#VecSimAlgo)
