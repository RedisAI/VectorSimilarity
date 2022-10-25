
/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "bm_utils.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"

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

    // HNSWParams is required to load v1 index
    HNSWParams params = {.type = VecSimType_FLOAT32,
                         .dim = BM_VecSimBasics::dim,
                         .metric = VecSimMetric_Cosine,
                         .multi = false,
                         .blockSize = BM_VecSimBasics::block_size};

    // Generate index from file.
    hnsw_index = HNSWFactory::NewIndex(GetSerializedIndexLocation(BM_VecSimBasics::hnsw_index_file),
                                       &params);

    auto hnsw_index_casted = reinterpret_cast<HNSWIndex<float, float> *>(hnsw_index);
    size_t ef_r = 10;
    hnsw_index_casted->setEf(ef_r);

    VecSimParams bf_params = {.algo = VecSimAlgo_BF,
                              .bfParams = BFParams{.type = VecSimType_FLOAT32,
                                                   .dim = BM_VecSimBasics::dim,
                                                   .metric = VecSimMetric_Cosine,
                                                   .initialCapacity = BM_VecSimBasics::n_vectors,
                                                   .blockSize = BM_VecSimBasics::block_size}};
    BM_VecSimBasics::bf_index = VecSimIndex_New(&bf_params);

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob = hnsw_index_casted->getDataByInternalId(i);
        VecSimIndex_AddVector(bf_index, blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    BM_VecSimBasics::queries = new std::vector<std::vector<float>>;
    load_test_vectors(BM_VecSimBasics::test_vectors_file, *queries, BM_VecSimBasics::n_queries,
                      BM_VecSimBasics::dim);
}

BM_VecSimBasics::~BM_VecSimBasics() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(hnsw_index);
        VecSimIndex_Free(bf_index);
        delete queries;
    }
}
