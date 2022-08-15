#include "bm_utils.h"
#include "VecSim/spaces/spaces.h"

void load_HNSW_index(const char *path, VecSimIndex *hnsw_index) {

    // Load the index file, if it exists in the expected path.
    auto location = std::string(getenv("ROOT"));
    auto file_name = location + "/" + path;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_ptr =
        reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex();
    auto serializer = hnswlib::HNSWIndexSerializer(hnsw_index_ptr);
    std::ifstream input(file_name, std::ios::binary);
    if (input.is_open()) {
        serializer.loadIndex(file_name, hnsw_index_ptr->GetDistFunc(), hnsw_index_ptr->GetDim());
        if (!serializer.checkIntegrity().valid_state) {
            throw std::runtime_error("The loaded HNSW index is corrupted. Exiting...");
        }
    } else {
        throw std::runtime_error("HNSW index file was not found in path. Exiting...");
    }
}

void load_test_vectors(const char *path, std::vector<std::vector<float>> &queries, size_t n_queries,
                       size_t dim) {
    auto location = std::string(std::string(getenv("ROOT")));
    auto file_name = location + "/" + path;

    std::ifstream input(file_name, std::ios::binary);

    queries.reserve(n_queries);
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
    size_t ef_r = 10;
    reinterpret_cast<HNSWIndex *>(hnsw_index)->setEf(ef_r);

    VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                              .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                   .dim = BM_VecSimBasics::dim,
                                                   .metric = VecSimMetric_Cosine,
                                                   .initialCapacity = BM_VecSimBasics::n_vectors,
                                                   .blockSize = BM_VecSimBasics::block_size}};
    BM_VecSimBasics::bf_index = VecSimIndex_New(&bf_params);

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob =
            reinterpret_cast<HNSWIndex *>(hnsw_index)->getHNSWIndex()->getDataByInternalId(i);
        VecSimIndex_AddVector(bf_index, blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    BM_VecSimBasics::queries = new std::vector<std::vector<float>>;
    load_test_vectors(BM_VecSimBasics::test_vectors_file, *queries, BM_VecSimBasics::n_queries,
                      BM_VecSimBasics::dim);
}

void BM_VecSimBasics::RunTopK_HNSW(benchmark::State &st, size_t ef, size_t iter, size_t k,
                                   size_t &correct, VecSimIndex *hnsw_index_,
                                   VecSimIndex *bf_index_) {
    auto query_params = VecSimQueryParams{.hnswRuntimeParams = HNSWRuntimeParams{.efRuntime = ef}};
    auto hnsw_results = VecSimIndex_TopKQuery(hnsw_index_, (*queries)[iter % n_queries].data(), k,
                                              &query_params, BY_SCORE);
    st.PauseTiming();

    // Measure recall:
    auto bf_results =
        VecSimIndex_TopKQuery(bf_index_, (*queries)[iter % n_queries].data(), k, nullptr, BY_SCORE);
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
        delete queries;
    }
}
