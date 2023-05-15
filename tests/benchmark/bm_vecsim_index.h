#pragma once

#include "bm_vecsim_general.h"
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
    static inline char *GetHNSWDataByInternalId(size_t id, unsigned short index_offset = 0) {
        return CastToHNSW(indices[VecSimAlgo_HNSWLIB + index_offset])->getDataByInternalId(id);
    }

private:
    static void Initialize();
    static void InsertToQueries(std::ifstream &input);
    static void loadTestVectors(const std::string &test_file, VecSimType type);
};

template <typename index_type_t>
size_t BM_VecSimIndex<index_type_t>::ref_count = 0;

// Needs to be explicitly initalized
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
    // dim, block_size, M, EF_C, n_veectors, is_multi, n_queries, hnsw_index_file and
    // test_queries_file are BM_VecSimGeneral static data members that are defined for a specific
    // index type benchmarks.
    BFParams bf_params = {.type = type,
                          .dim = dim,
                          .metric = VecSimMetric_Cosine,
                          .multi = is_multi,
                          .initialCapacity = n_vectors,
                          .blockSize = block_size};

    indices.push_back(CreateNewIndex(bf_params));

    HNSWParams params = {.type = type,
                         .dim = dim,
                         .metric = VecSimMetric_Cosine,
                         .multi = is_multi,
                         .initialCapacity = n_vectors,
                         .blockSize = block_size,
                         .M = M,
                         .efConstruction = EF_C};

    // Initialize and load HNSW index for DBPedia data set.
    indices.push_back(HNSWFactory::NewIndex(AttachRootPath(hnsw_index_file), &params));

    auto *hnsw_index = CastToHNSW(indices[VecSimAlgo_HNSWLIB]);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);

    // Add the same vectors to Flat index.
    for (size_t i = 0; i < n_vectors; ++i) {
        char *blob = GetHNSWDataByInternalId(i);
        // Fot multi value indices, the internal id is not necessarily equal the label.
        size_t label = CastToHNSW(indices[VecSimAlgo_HNSWLIB])->getExternalLabel(i);
        VecSimIndex_AddVector(indices[VecSimAlgo_BF], blob, label);
    }

    // Load the test query vectors form file. Index file path is relative to repository root dir.
    loadTestVectors(AttachRootPath(test_queries_file), type);
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
    for (size_t i = 0; i < n_queries; i++) {
        std::vector<data_t> query(dim);
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}
