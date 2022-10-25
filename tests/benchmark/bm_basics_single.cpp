#include "bm_utils.h"

template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};
using fp32_index_t = IndexType<VecSimType_FLOAT32, float, float>;
using fp64_index_t = IndexType<VecSimType_FLOAT64, double, double>;
template<>
size_t BM_VecSimBasics<false>::n_vectors = 1000000;
template<>
size_t BM_VecSimBasics<false>::n_queries = 10000;
template<>
size_t BM_VecSimBasics<false>::dim = 768;
template<>
size_t BM_VecSimBasics<false>::M = 64;
template<>
size_t BM_VecSimBasics<false>::EF_C = 512;
template<>
size_t BM_VecSimBasics<false>::block_size = 1024;
template<>
const char *BM_VecSimBasics<false>::hnsw_index_file =
    "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1";
template<>
const char *BM_VecSimBasics<false>::test_vectors_file =
    "tests/benchmark/data/DBpedia-test_vectors-n10k.raw";
template<>
size_t BM_VecSimBasics<false>::ref_count = 0;



template <typename index_type_t>
class BM_VecSimBasics_Single : public BM_VecSimBasics<false> {
public:
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;
    
    BM_VecSimBasics_Single() : BM_VecSimBasics<false>(index_type_t::get_index_type()) {};
    
    static std::vector<std::vector<data_t>> queries;

    void AddVector(benchmark::State &st);
    template <typename algo_t>
    void DeleteVector(algo_t index, benchmark::State &st);
    static std::vector<VecSimIndex *> indices;

    virtual void InitializeIndicesVector(VecSimIndex *bf_index, VecSimIndex *hnsw_index) override;
    virtual void InsertToQueries(std::ifstream& input) override;
    virtual void  LoadHNSWIndex(std::string location) override;
    virtual inline char *GetHNSWDataByInternalId(size_t id) const override {
        return CastToHNSW(indices[VecSimAlgo_HNSWLIB])->getDataByInternalId(id);
    }
    virtual inline VecSimIndex *GetBF() override {
        return indices[VecSimAlgo_BF];
    }

    ~BM_VecSimBasics_Single();
private:
    HNSWIndex<data_t, dist_t> *CastToHNSW(VecSimIndex *index) const {
        return reinterpret_cast<HNSWIndex<data_t, dist_t> *>(index);
    }

    inline std::vector<data_t> NewQuery() {
        std::vector<data_t> ret(dim);
        return ret;
    }

};

template <>
std::vector<VecSimIndex *> BM_VecSimBasics_Single<fp32_index_t>::indices = std::vector<VecSimIndex *>();
template <>
std::vector<VecSimIndex *> BM_VecSimBasics_Single<fp64_index_t>::indices = std::vector<VecSimIndex *>();
template <>
std::vector<std::vector<float>> BM_VecSimBasics_Single<fp32_index_t>::queries = std::vector<std::vector<float>>();
template <>
std::vector<std::vector<double>> BM_VecSimBasics_Single<fp64_index_t>::queries = std::vector<std::vector<double>>();


BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, AddVector_fp32, fp32_index_t)(benchmark::State &st) {
    AddVector(st);
}

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimBasics_Single, AddVector_fp64, fp64_index_t)(benchmark::State &st) {
    AddVector(st);
}

#define UNIT_AND_ITERATIONS Unit(benchmark::kMillisecond) \
    ->Iterations((long)BM_VecSimBasics<false>::block_size)

BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, AddVector_fp32)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)->Arg(VecSimAlgo_HNSWLIB);
BENCHMARK_REGISTER_F(BM_VecSimBasics_Single, AddVector_fp64)
    ->UNIT_AND_ITERATIONS->Arg(VecSimAlgo_BF)->Arg(VecSimAlgo_HNSWLIB);

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::InitializeIndicesVector(VecSimIndex *bf_index, VecSimIndex *hnsw_index) {
    indices.push_back(bf_index);
    indices.push_back(hnsw_index);
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::InsertToQueries(std::ifstream& input) {
    for (size_t i = 0; i < BM_VecSimBasics::n_queries; i++) {
        auto query = NewQuery();
        input.read((char *)query.data(), dim * sizeof(data_t));
        queries.push_back(query);
    }
}

template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::LoadHNSWIndex(std::string location){
    auto *hnsw_index = CastToHNSW(indices[VecSimAlgo_HNSWLIB]);
    hnsw_index->loadIndex(location);
    size_t ef_r = 10;
    hnsw_index->setEf(ef_r);

}
template <typename index_type_t>
void BM_VecSimBasics_Single<index_type_t>::AddVector(benchmark::State &st) {
    // Add a new vector from the test vectors in every iteration.
    size_t iter = 0;
    size_t new_id = VecSimIndex_IndexSize(indices[st.range(0)]);
    size_t memory_delta = 0;
    for (auto _ : st) {
        memory_delta +=
            VecSimIndex_AddVector(indices[st.range(0)], queries[iter % n_queries].data(), new_id++);
        iter++;
    }
    st.counters["memory"] = (double)memory_delta / (double)iter;

    // Clean-up.
    size_t new_index_size = VecSimIndex_IndexSize(indices[st.range(0)]);
    for (size_t id = n_vectors; id < new_index_size; id++) {
        VecSimIndex_DeleteVector(indices[st.range(0)], id);
    }       
}

template <typename index_type_t>
template <typename algo_t>
void BM_VecSimBasics_Single<index_type_t>::DeleteVector(algo_t index, benchmark::State &st) {
     // Remove a different vector in every execution.
    std::vector<std::vector<data_t>> blobs;
    size_t id_to_remove = 0;
    double memory_delta = 0;
    size_t iter = 0;


    for (auto _ : st) {
        st.PauseTiming();
        auto removed_vec = std::vector<data_t>(dim);
        memcpy(removed_vec.data(), index->getDataByInternalId(id_to_remove),
               dim * sizeof(data_t));
        blobs.push_back(removed_vec);
        st.ResumeTiming();

        iter++;
        auto delta = (double)VecSimIndex_DeleteVector(index, id_to_remove++);
        memory_delta += delta;
    }
    st.counters["memory"] = memory_delta / (double)iter;

    // Restore index state.
    for (size_t i = 0; i < blobs.size(); i++) {
        VecSimIndex_AddVector(index, blobs[i].data(), i);
    }
}

template <typename index_type_t>
BM_VecSimBasics_Single<index_type_t>::~BM_VecSimBasics_Single() {
    ref_count--;
    if (ref_count == 0) {
        VecSimIndex_Free(indices[VecSimAlgo_BF]);
        VecSimIndex_Free(indices[VecSimAlgo_HNSWLIB]);
    }
}

BENCHMARK_MAIN();
