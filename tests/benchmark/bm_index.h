#pragma once

#include "bm_utils.h"
template <typename index_type_t>
class BM_VecSimIndex : public BM_VecSimGeneral {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;

    static size_t ref_count;

    static std::vector<std::vector<data_t>> queries;

    static std::vector<VecSimIndex *> indices;

    BM_VecSimIndex();

    virtual ~BM_VecSimIndex();

protected:
    static inline HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }
    static inline char *GetHNSWDataByInternalId(size_t id, Offset_t index_offset = 0) {
        return CastToHNSW(indices[VecSimAlgo_HNSWLIB + index_offset])->getDataByInternalId(id);
    }

    template <typename IndexParams>
    static inline VecSimIndex *CreateNewIndex(IndexParams &index_params) {
        VecSimParams params = CreateParams(index_params);
        return VecSimIndex_New(&params);
    }
    static void loadTestVectors(const std::string &test_file, VecSimType type);

    static inline std::string AttachRootPath(std::string file_name) {
        return std::string(getenv("ROOT")) + "/" + file_name;
    }

private:
    static void Initialize();
    static void InsertToQueries(std::ifstream &input);
};

template <typename index_type_t>
size_t BM_VecSimIndex<index_type_t>::ref_count = 0;

// Needs to be explicitly intialized
template <>
std::vector<std::vector<float>> BM_VecSimIndex<fp32_index_t>::queries{};

template <>
std::vector<std::vector<double>> BM_VecSimIndex<fp64_index_t>::queries{};

template <>
std::vector<VecSimIndex *> BM_VecSimIndex<fp32_index_t>::indices{};
template <>
std::vector<VecSimIndex *> BM_VecSimIndex<fp64_index_t>::indices{};

template <typename index_type_t>
BM_VecSimIndex<index_type_t>::~BM_VecSimIndex() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(indices[VecSimAlgo_BF]);
        VecSimIndex_Free(indices[VecSimAlgo_HNSWLIB]);
    }
}

template <typename index_type_t>
BM_VecSimIndex<index_type_t>::BM_VecSimIndex() {
    if (ref_count == 0) {
        // Initialize the static members.
        Initialize();
    }
    ref_count++;
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::Initialize() {

    VecSimType type = index_type_t::get_index_type();

    BFParams bf_params = {.type = type,
                          .dim = dim,
                          .metric = VecSimMetric_Cosine,
                          .multi = IS_MULTI,
                          .initialCapacity = n_vectors,
                          .blockSize = block_size};

    indices.push_back(CreateNewIndex(bf_params));

    HNSWParams params = {.type = type,
                         .dim = DIM,
                         .metric = VecSimMetric_Cosine,
                         .multi = IS_MULTI,
                         .initialCapacity = N_VECTORS,
                         .blockSize = BM_VecSimGeneral::block_size,
                         .M = BM_VecSimGeneral::M,
                         .efConstruction = BM_VecSimGeneral::EF_C};

    // Initialize and load HNSW index for DBPedia data set.
    indices.push_back(
        HNSWFactory::NewIndex(AttachRootPath(BM_VecSimGeneral::hnsw_index_file), &params));

    auto *hnsw_index = CastToHNSW(indices[VecSimAlgo_HNSWLIB]);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        // char *blob = GetHNSWDataByInternalId(i);
        char *blob = CastToHNSW(indices[VecSimAlgo_HNSWLIB])->getDataByInternalId(i);
        VecSimIndex_AddVector(indices[VecSimAlgo_BF], blob, i);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    loadTestVectors(AttachRootPath(BM_VecSimGeneral::test_queries_file), type);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::loadTestVectors(const std::string &test_file, VecSimType type) {

    std::ifstream input(test_file, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Test vectors file was not found in path. Exiting...");
    }
    input.seekg(0, std::ifstream::beg);

    InsertToQueries(input);
}

template <typename index_type_t>
void BM_VecSimIndex<index_type_t>::InsertToQueries(std::ifstream &input) {
    for (size_t i = 0; i < N_QUERIES; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}
