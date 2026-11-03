/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "bm_spaces.h"
#include <memory>
#include <string>
#include "VecSim/spaces/computer/preprocessors.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/types/sq4.h"

// Compares the SQ4 and SQ8 distance kernels on production-preprocessed blobs: stored-to-query
// (asymmetric, FP32 query) and stored-to-stored (symmetric), for IP and L2. For each format the
// scalar kernel and the function selected by the production dispatcher are measured. SQ4 has only
// scalar kernels, so its dispatched function is expected to be the scalar one; the label reports
// which of the two the dispatcher returned.

namespace {

template <typename StorageType>
constexpr bool is_sq4 = std::is_same_v<StorageType, vecsim_types::sq4>;

template <typename StorageType>
spaces::dist_func_t<float> scalarKernel(VecSimMetric metric, bool symmetric) {
    const bool ip = metric == VecSimMetric_IP;
    if constexpr (is_sq4<StorageType>) {
        if (symmetric)
            return ip ? SQ4_SQ4_InnerProduct : SQ4_SQ4_L2Sqr;
        return ip ? SQ4_FP32_InnerProduct : SQ4_FP32_L2Sqr;
    } else {
        if (symmetric)
            return ip ? SQ8_SQ8_InnerProduct : SQ8_SQ8_L2Sqr;
        return ip ? SQ8_FP32_InnerProduct : SQ8_FP32_L2Sqr;
    }
}

} // namespace

template <typename StorageType>
class BM_VecSimSpaces_Quantized : public benchmark::Fixture {
protected:
    std::shared_ptr<VecSimAllocator> allocator;
    size_t dim;
    MemoryUtils::unique_blob stored1, stored2, query2;
    spaces::dist_func_t<float> asymmetric, symmetric;
    spaces::dist_func_t<float> scalar_asymmetric, scalar_symmetric;

    template <VecSimMetric Metric>
    void prepare(const std::vector<float> &v1, const std::vector<float> &v2) {
        unsigned char storage_alignment = 0, symmetric_alignment = 0, query_alignment = 0;
        asymmetric =
            spaces::GetDistFunc<StorageType, float, float>(Metric, dim, &storage_alignment);
        symmetric = spaces::GetDistFunc<StorageType, float>(Metric, dim, &symmetric_alignment);
        storage_alignment = spaces::combineAlignments(storage_alignment, symmetric_alignment);
        spaces::GetDistFunc<float, float>(Metric, dim, &query_alignment);

        void *s1 = nullptr, *s2 = nullptr, *q2 = nullptr;
        size_t s1_size = 0, s2_size = 0, q2_size = 0;
        QuantPreprocessor<float, Metric, false, StorageType> pp(allocator, dim);
        pp.preprocessForStorage(v1.data(), s1, s1_size, storage_alignment);
        pp.preprocessForStorage(v2.data(), s2, s2_size, storage_alignment);
        pp.preprocessQuery(v2.data(), q2, q2_size, query_alignment);
        auto deleter = [alloc = allocator](void *ptr) { alloc->free_allocation(ptr); };
        stored1 = MemoryUtils::unique_blob(s1, deleter);
        stored2 = MemoryUtils::unique_blob(s2, deleter);
        query2 = MemoryUtils::unique_blob(q2, deleter);
    }

    void run(benchmark::State &st, bool use_scalar, bool use_symmetric) {
        auto function = use_symmetric ? (use_scalar ? scalar_symmetric : symmetric)
                                      : (use_scalar ? scalar_asymmetric : asymmetric);
        const void *right = use_symmetric ? stored2.get() : query2.get();
        for (auto _ : st)
            benchmark::DoNotOptimize(function(stored1.get(), right, dim));
        const bool is_scalar = function == (use_symmetric ? scalar_symmetric : scalar_asymmetric);
        st.SetLabel(std::string("isa=") + (is_scalar ? "scalar" : "simd"));
    }

public:
    void SetUp(const benchmark::State &state) override {
        allocator = VecSimAllocator::newVecsimAllocator();
        dim = state.range(0);
        std::mt19937 rng(47);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        std::vector<float> v1(dim), v2(dim);
        for (size_t d = 0; d < dim; ++d) {
            v1[d] = uniform(rng);
            v2[d] = uniform(rng);
        }
        const auto metric = static_cast<VecSimMetric>(state.range(1));
        if (metric == VecSimMetric_IP)
            prepare<VecSimMetric_IP>(v1, v2);
        else
            prepare<VecSimMetric_L2>(v1, v2);
        scalar_asymmetric = scalarKernel<StorageType>(metric, false);
        scalar_symmetric = scalarKernel<StorageType>(metric, true);
    }
    void TearDown(const benchmark::State &) override {
        stored1.reset();
        stored2.reset();
        query2.reset();
        allocator.reset();
    }
};

using BM_VecSimSpaces_SQ4 = BM_VecSimSpaces_Quantized<vecsim_types::sq4>;
using BM_VecSimSpaces_SQ8 = BM_VecSimSpaces_Quantized<vecsim_types::sq8>;

#define REGISTER_QUANTIZED_DISTANCE(fixture, name, use_scalar, use_symmetric)                      \
    BENCHMARK_DEFINE_F(fixture, name)(benchmark::State & st) {                                     \
        run(st, use_scalar, use_symmetric);                                                        \
    }                                                                                              \
    BENCHMARK_REGISTER_F(fixture, name)                                                            \
        ->ArgsProduct({{15, 16, 17, 767, 768}, {VecSimMetric_IP, VecSimMetric_L2}})                \
        ->ArgNames({"Dimension", "Metric"})                                                        \
        ->MinTime(0.05)                                                                            \
        ->Unit(benchmark::kNanosecond)

#define REGISTER_QUANTIZED_SET(fixture)                                                            \
    REGISTER_QUANTIZED_DISTANCE(fixture, Asymmetric_Scalar, true, false);                          \
    REGISTER_QUANTIZED_DISTANCE(fixture, Asymmetric_Dispatch, false, false);                       \
    REGISTER_QUANTIZED_DISTANCE(fixture, Symmetric_Scalar, true, true);                            \
    REGISTER_QUANTIZED_DISTANCE(fixture, Symmetric_Dispatch, false, true)

REGISTER_QUANTIZED_SET(BM_VecSimSpaces_SQ4);
REGISTER_QUANTIZED_SET(BM_VecSimSpaces_SQ8);

BENCHMARK_MAIN();
