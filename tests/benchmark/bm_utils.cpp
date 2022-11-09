#include "bm_utils.h"

void load_HNSW_index(const char *path, VecSimIndex *hnsw_index) {

    // Load the index file, if it exists in the expected path.
    auto location = std::string(getenv("ROOT"));
    auto file_name = location + "/" + path;
    auto serializer = HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index));
    std::ifstream input(file_name, std::ios::binary);
    if (input.is_open()) {
        serializer.loadIndex(file_name);
        if (!serializer.checkIntegrity().valid_state) {
            throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
        }
    } else {
        throw std::runtime_error("HNSW index file was not found in path. Exiting...");
    }
}

std::vector<std::vector<float>> load_test_vectors(const char *path, size_t n_queries, size_t dim) {
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/" + path;

    std::ifstream input(file_name, std::ios::binary);

    std::vector<std::vector<float>> queries(n_queries);
    if (input.is_open()) {
        input.seekg(0, std::ifstream::beg);
        for (size_t i = 0; i < n_queries; i++) {
            std::vector<float> query(dim);
            input.read((char *)query.data(), dim * sizeof(float));
            queries[i] = query;
        }
    } else {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    return queries;
}

BM_VecSimBasics::BM_VecSimBasics() {
    if (ref_count == 0) {
        // Initialize the static members.
        Initialize();
    }
    ref_count++;
}

void BM_VecSimBasics::Initialize() {

    // Initialize and load HNSW index for DBPedia data set.
    VecSimParams params = {.algo = VecSimAlgo_HNSWLIB,
                           .hnswParams = HNSWParams{.type = VecSimType_FLOAT32,
                                                    .dim = BM_VecSimBasics::dim,
                                                    .metric = VecSimMetric_Cosine,
                                                    .initialCapacity = BM_VecSimBasics::n_vectors,
                                                    .blockSize = BM_VecSimBasics::block_size,
                                                    .M = BM_VecSimBasics::M,
                                                    .efConstruction = BM_VecSimBasics::EF_C}};
    BM_VecSimBasics::hnsw_index = VecSimIndex_New(&params);

    // Load pre-generated HNSW index. Index file path is relative to repository root dir.
    load_HNSW_index(hnsw_index_file, hnsw_index);

#ifdef PATTERN
    printf("USING PATTERN %d\n", PATTERN);
#if PATTERN == 1
    for (size_t i = 0; i < n_vectors / 2; ++i) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#elif PATTERN == 2
    for (size_t i = 0; i < n_vectors / 4; ++i) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#elif PATTERN == 3
    for (size_t i = 0; i < n_vectors / 10; ++i) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#elif PATTERN == 4
    for (size_t i = n_vectors / 2; i < n_vectors; ++i) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#elif PATTERN == 5
    for (size_t i = n_vectors - (n_vectors / 4); i < n_vectors; ++i) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#elif PATTERN == 6
    for (size_t i = n_vectors - (n_vectors / 10); i < n_vectors; ++i) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#elif PATTERN == 7
    for (size_t i = 0; i < n_vectors; i+=2) {
        reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->markDelete(i);
    }
#else
    printf("Unknown pattern. running on entire data.\n");
#endif
#endif

    size_t ef_r = 10;
    reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->setEf(ef_r);

    VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                              .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                   .dim = BM_VecSimBasics::dim,
                                                   .metric = VecSimMetric_Cosine,
                                                   .initialCapacity = BM_VecSimBasics::n_vectors,
                                                   .blockSize = BM_VecSimBasics::block_size}};
    BM_VecSimBasics::bf_index = VecSimIndex_New(&bf_params);

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        if (!reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->isMarkedDeleted(i)) {
            char *blob =
                reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index)->getDataByInternalId(i);
            VecSimIndex_AddVector(bf_index, blob, i);
        }
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    BM_VecSimBasics::queries = load_test_vectors(BM_VecSimBasics::test_vectors_file,
                                                 BM_VecSimBasics::n_queries, BM_VecSimBasics::dim);
}

void BM_VecSimBasics::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                                   size_t &correct, VecSimIndex *hnsw_index_,
                                   VecSimIndex *bf_index_) {
    auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index_, queries[iter % n_queries].data(), k,
                                              &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results =
        VecSimIndex_TopKQuery(bf_index_, queries[iter % n_queries].data(), k, nullptr, BY_SCORE);
    auto hnsw_it = VecSimQueryResult_List_GetIterator(hnsw_results);
    while (VecSimQueryResult_IteratorHasNext(hnsw_it)) {
        auto hnsw_res_item = VecSimQueryResult_IteratorNext(hnsw_it);
        auto bf_it = VecSimQueryResult_List_GetIterator(bf_results);
        while (VecSimQueryResult_IteratorHasNext(bf_it)) {
            auto bf_res_item = VecSimQueryResult_IteratorNext(bf_it);
            if (VecSimQueryResult_GetId(hnsw_res_item) == VecSimQueryResult_GetId(bf_res_item)) {
                correct++;
                break;
            }
        }
        VecSimQueryResult_IteratorFree(bf_it);
    }
    VecSimQueryResult_IteratorFree(hnsw_it);

    VecSimQueryResult_Free(bf_results);
    VecSimQueryResult_Free(hnsw_results);
    st.ResumeTiming();
}

BM_VecSimBasics::~BM_VecSimBasics() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(hnsw_index);
        VecSimIndex_Free(bf_index);
    }
}
