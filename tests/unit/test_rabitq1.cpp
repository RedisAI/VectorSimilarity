/* Copyright (c) 2006-Present, Redis Ltd. All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3. */
#include "gtest/gtest.h"
#include "VecSim/quantization/rabitq_research.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/vec_sim.h"
#include <rabitqlib/core/detail/simd/space_dispatch.hpp>
#include <rabitqlib/utils/cpu_features.hpp>
#include <rabitqlib/utils/space.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

using namespace vecsim::experimental;

namespace {
std::vector<float> unitVector(size_t dim, size_t seed) {
    std::vector<float> x(dim);
    double norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        x[i] = std::sin(double(i + 1) * (seed + 1) * 0.713) +
               0.3 * std::cos(double(i + 3) * (seed + 2) * 0.419);
        norm += double(x[i]) * x[i];
    }
    for (float &v : x)
        v = static_cast<float>(v / std::sqrt(norm));
    return x;
}

// Use the official lossless conversion API, independently of logicalCode().
std::vector<unsigned> unpackedCodes(const RaBitQContext &c, const void *record,
                                   const std::shared_ptr<VecSimAllocator> &allocator) {
    const auto r = rabitq_api::requirements(c.parameters(), rabitq_api::CodeLayout::UnpackedU16);
    auto scratch = allocator->allocate_aligned_unique(r.workspace_bytes, r.workspace_alignment);
    std::vector<uint8_t> bytes(r.code_bytes);
    rabitq_api::convert_codes(c.parameters(), c.codeView(record),
        {rabitq_api::CodeLayout::UnpackedU16, {bytes.data(), bytes.size()}, {}},
        {scratch.get(), r.workspace_bytes});
    std::vector<unsigned> result(c.paddedDim());
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = bytes[2 * i] | unsigned(bytes[2 * i + 1]) << 8;
    return result;
}

std::vector<double> decodedReference(const RaBitQContext &c, const void *record,
                                     const std::shared_ptr<VecSimAllocator> &allocator) {
    const auto codes = unpackedCodes(c, record, allocator);
    const auto f = c.factors(record);
    std::vector<double> result(c.paddedDim());
    const double midpoint = double((1U << c.bits()) - 1) / 2;
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(c.logicalCode(record, i), codes[i]);
        result[i] = (c.rotatedMean().empty() ? 0 : c.rotatedMean()[i]) -
                    double(f.rescale) * (double(codes[i]) - midpoint);
    }
    return result;
}

double queryReference(const RaBitQContext &c, const void *record, const float *rotated_query,
                      const std::shared_ptr<VecSimAllocator> &allocator) {
    const auto decoded = decodedReference(c, record, allocator);
    double ip = 0;
    for (size_t i = 0; i < decoded.size(); ++i)
        ip += decoded[i] * rotated_query[i];
    return 1 - ip + (double(c.factors(record).add) - 1);
}

void numericCase(size_t dim, size_t bits, bool centered) {
    SCOPED_TRACE(::testing::Message() << "dim=" << dim << " bits=" << bits << " mean=" << centered);
    auto allocator = VecSimAllocator::newVecsimAllocator();
    auto x = unitVector(dim, 1), y = unitVector(dim, 3), q = unitVector(dim, 7);
    std::vector<float> mean(centered ? dim : 0);
    for (size_t i = 0; i < mean.size(); ++i)
        mean[i] = 0.4F * x[i] + 0.2F * y[i];
    auto parts = CreateRaBitQComponents(allocator, dim, bits, mean);
    const auto &c = *parts.context;
    auto a = parts.preprocessors->preprocessForStorage(x.data(), dim * sizeof(float));
    auto b = parts.preprocessors->preprocessForStorage(y.data(), dim * sizeof(float));
    auto query = parts.preprocessors->preprocessQuery(q.data(), dim * sizeof(float));
    EXPECT_EQ(c.paddedDim(), (dim + 31) / 32 * 32);
    EXPECT_EQ(c.requirements().code_bytes, c.paddedDim() / 8);
    EXPECT_EQ(c.requirements().extra_bytes, c.paddedDim() * (bits - 1) / 8);
    EXPECT_EQ(c.storageBytes(), c.paddedDim() * bits / 8 + 3 * sizeof(float));
    if (bits == 1)
        EXPECT_EQ(c.codeView(a.get()).extra.data, nullptr);

    const auto da = decodedReference(c, a.get(), allocator);
    const auto db = decodedReference(c, b.get(), allocator);
    std::vector<float> decoded(c.paddedDim());
    c.decode(a.get(), decoded);
    double pair_ip = 0;
    for (size_t i = 0; i < decoded.size(); ++i) {
        EXPECT_EQ(decoded[i], static_cast<float>(da[i]));
        pair_ip += da[i] * db[i];
    }
    const auto fa = c.factors(a.get()), fb = c.factors(b.get());
    const double pair = 1 - pair_ip + (double(fa.add) - 1) + (double(fb.add) - 1);
    EXPECT_NEAR(c.storedDistance(a.get(), b.get()), pair, 2e-7);
    EXPECT_NEAR(c.storedDistance(b.get(), a.get()), pair, 2e-7);
    EXPECT_FLOAT_EQ(c.storedDistance(a.get(), b.get()), c.storedDistance(b.get(), a.get()));
    for (const void *record : {a.get(), b.get()}) {
        const double expected = queryReference(c, record, c.queryVector(query.get()), allocator);
        // FP32 reductions operate on normalized vectors, so distances are O(1).
        EXPECT_NEAR(c.queryDistance(record, query.get()), expected, 8e-6);
        EXPECT_FLOAT_EQ(c.queryDistance(record, query.get()),
            rabitq_api::compute(c.queryState(query.get()), c.codeView(record), c.factors(record)));
        EXPECT_FLOAT_EQ(parts.calculator->getDistanceDispatch(DistanceMode::StoredToQuery)(
            record, query.get(), dim), c.queryDistance(record, query.get()));
    }
    EXPECT_FLOAT_EQ(parts.calculator->getDistanceDispatch(DistanceMode::StoredToStored)(
        a.get(), b.get(), dim), c.storedDistance(a.get(), b.get()));

    // Encode directly with the actual centroid: all code and factor bytes must
    // match, without subtracting a centroid externally or replacing its bias.
    std::vector<float> rotated(c.paddedDim());
    c.rotate(x.data(), rotated.data());
    if (bits == 1) {
        double norm = 0, residual_code_ip = 0, mean_code_ip = 0, residual_mean_ip = 0;
        for (size_t i = 0; i < c.paddedDim(); ++i) {
            const double m = c.rotatedMean().empty() ? 0 : c.rotatedMean()[i];
            const float residual = rotated[i] - static_cast<float>(m);
            const double sign = residual > 0 ? 0.5 : -0.5;
            EXPECT_EQ(c.logicalCode(a.get(), i), unsigned(residual > 0));
            norm += double(residual) * residual;
            residual_code_ip += residual * sign;
            mean_code_ip += m * sign;
            residual_mean_ip += residual * m;
        }
        ASSERT_GT(residual_code_ip, 0);
        const double expected_add = 1 - residual_mean_ip + norm * mean_code_ip / residual_code_ip;
        const double expected_scale = -norm / residual_code_ip;
        const double expected_error = std::sqrt(norm) * 1.9 * std::sqrt(std::max(0.0,
            (norm * (0.25 * c.paddedDim()) / (residual_code_ip * residual_code_ip) - 1) /
            (c.paddedDim() - 1)));
        EXPECT_NEAR(fa.add, expected_add, 2e-6);
        EXPECT_NEAR(fa.rescale, expected_scale, 2e-6);
        EXPECT_NEAR(fa.error, expected_error, 2e-6);
    }
    const auto r = c.requirements();
    auto scratch = allocator->allocate_aligned_unique(r.workspace_bytes, r.workspace_alignment);
    std::vector<uint8_t> official(c.storageBytes());
    auto f = rabitq_api::encode(c.parameters(), rotated.data(),
        c.rotatedMean().empty() ? nullptr : c.rotatedMean().data(),
        {rabitq_api::CodeLayout::SplitPacked, {official.data(), r.code_bytes},
         {r.extra_bytes ? official.data() + r.code_bytes : nullptr, r.extra_bytes}},
        {scratch.get(), r.workspace_bytes});
    EXPECT_EQ(std::memcmp(official.data(), a.get(), r.code_bytes + r.extra_bytes), 0);
    EXPECT_EQ(fa.add, f.add);
    EXPECT_EQ(fa.rescale, f.rescale);
    EXPECT_EQ(fa.error, f.error);
    float factor_bytes[3];
    std::memcpy(factor_bytes, static_cast<const uint8_t *>(a.get()) + c.storageBytes() - 12, 12);
    EXPECT_EQ(factor_bytes[0], f.add);
    EXPECT_EQ(factor_bytes[1], f.rescale);
    EXPECT_EQ(factor_bytes[2], f.error);
}
} // namespace

TEST(RaBitQ1, OfficialEncoderIndependentQueryAndSymmetricGraphReferences) {
    for (size_t dim : {65, 97, 768})
        for (size_t bits : {1, 2, 3})
            for (bool centered : {false, true})
                numericCase(dim, bits, centered);
}

TEST(RaBitQ1, PaddedRotationPreservesInnerProducts) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    for (size_t dim : {65, 97, 768}) {
        auto parts = CreateRaBitQComponents(allocator, dim, 1);
        const auto &c = *parts.context;
        auto x = unitVector(dim, 12), y = unitVector(dim, 17);
        std::vector<float> rx(c.paddedDim()), ry(c.paddedDim());
        c.rotate(x.data(), rx.data());
        c.rotate(y.data(), ry.data());
        double xx = 0, rxx = 0, xy = 0, rxy = 0;
        for (size_t i = 0; i < dim; ++i) {
            xx += double(x[i]) * x[i];
            xy += double(x[i]) * y[i];
        }
        for (size_t i = 0; i < c.paddedDim(); ++i) {
            rxx += double(rx[i]) * rx[i];
            rxy += double(rx[i]) * ry[i];
        }
        EXPECT_NEAR(rxx, xx, 2e-6);
        EXPECT_NEAR(rxy, xy, 2e-6);
    }
}

TEST(RaBitQ1, SignOnlyNativeAndScalarScoresUseUnalignedRecordsAndQueries) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    for (size_t dim : {65, 97, 768}) {
        auto parts = CreateRaBitQComponents(allocator, dim, 1);
        const auto &c = *parts.context;
        const auto r = c.requirements();
        auto scratch = allocator->allocate_aligned_unique(r.workspace_bytes, r.workspace_alignment);
        // Float-aligned, deliberately not SIMD-aligned query; byte-unaligned record.
        std::vector<float> query_storage(c.paddedDim() + 8);
        float *query = query_storage.data();
        while (reinterpret_cast<uintptr_t>(query) % 16 != 4) ++query;
        const auto values = unitVector(c.paddedDim(), 11);
        std::copy(values.begin(), values.end(), query);
        for (size_t pattern = 0; pattern < 5; ++pattern) {
            std::vector<uint8_t> integers(c.paddedDim() * 2, 0);
            for (size_t i = 0; i < c.paddedDim(); ++i)
                integers[2 * i] = pattern < 2 ? pattern : ((13 * i + i / 7 + pattern) & 1);
            std::vector<uint8_t> storage(c.storageBytes() + 9, 0xAD);
            for (size_t offset = 1; offset <= 8; ++offset) {
                std::fill(storage.begin(), storage.end(), 0xAD);
                auto *record = storage.data() + offset;
                rabitq_api::convert_codes(c.parameters(),
                    {rabitq_api::CodeLayout::UnpackedU16, {integers.data(), integers.size()}, {}},
                    {rabitq_api::CodeLayout::SplitPacked, {record, r.code_bytes}, {}},
                    {scratch.get(), r.workspace_bytes});
                const rabitq_api::Factors f{0.73F, -0.21F, 0.08F};
                std::memcpy(record + r.code_bytes, &f.add, 4);
                std::memcpy(record + r.code_bytes + 4, &f.rescale, 4);
                std::memcpy(record + r.code_bytes + 8, &f.error, 4);
                const auto before = storage;
                double raw_reference = 0, query_sum = 0;
                for (size_t i = 0; i < c.paddedDim(); ++i) {
                    EXPECT_EQ(c.logicalCode(record, i), integers[2 * i]);
                    raw_reference += double(query[i]) * integers[2 * i];
                    query_sum += query[i];
                }
                const float scalar = rabitqlib::simd::mask_ip_x0_q_generic(query, record, c.paddedDim());
                const float selected = rabitqlib::mask_ip_x0_q(query, record, c.paddedDim());
                EXPECT_EQ(scalar, static_cast<float>(raw_reference));
                EXPECT_NEAR(selected, raw_reference, 2e-6);
#if defined(__aarch64__) || defined(_M_ARM64)
                ASSERT_TRUE(rabitqlib::cpu::has_neon());
                EXPECT_EQ(selected, rabitqlib::simd::mask_ip_x0_q_neon(query, record, c.paddedDim()));
#endif
                const auto state = rabitq_api::prepare_query(c.parameters(), query, nullptr);
                const double corrected = f.add + double(f.rescale) * (raw_reference - 0.5 * query_sum);
                EXPECT_NEAR(rabitq_api::compute(state, c.codeView(record), f), corrected, 3e-6);
                EXPECT_EQ(storage, before);
                for (size_t i = 0; i < offset; ++i) EXPECT_EQ(storage[i], 0xAD);
                for (size_t i = offset + c.storageBytes(); i < storage.size(); ++i)
                    EXPECT_EQ(storage[i], 0xAD);
                c.encodeRotated(values.data(), record, {scratch.get(), r.workspace_bytes});
                EXPECT_EQ(decodedReference(c, record, allocator).size(), c.paddedDim());
                for (size_t i = 0; i < offset; ++i) EXPECT_EQ(storage[i], 0xAD);
                for (size_t i = offset + c.storageBytes(); i < storage.size(); ++i)
                    EXPECT_EQ(storage[i], 0xAD);
            }
        }
    }
}

TEST(RaBitQ1, ZeroInputZeroResidualAndCentroidBiasRemainObservable) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    for (size_t dim : {65, 97, 768}) {
        std::vector<float> zero(dim);
        auto no_mean = CreateRaBitQComponents(allocator, dim, 1);
        auto z = no_mean.preprocessors->preprocessForStorage(zero.data(), dim * 4);
        auto zq = no_mean.preprocessors->preprocessQuery(zero.data(), dim * 4);
        EXPECT_EQ(no_mean.context->factors(z.get()).rescale, 0);
        EXPECT_EQ(no_mean.context->factors(z.get()).add, 1);
        EXPECT_EQ(no_mean.context->queryDistance(z.get(), zq.get()), 1);
        EXPECT_EQ(no_mean.context->storedDistance(z.get(), z.get()), 1);
        for (size_t i = 0; i < no_mean.context->paddedDim(); ++i)
            EXPECT_EQ(no_mean.context->logicalCode(z.get(), i), 0);
        auto mean = unitVector(dim, 1), x = unitVector(dim, 5), q = unitVector(dim, 8);
        for (float &v : mean) v *= 0.5F;
        auto parts = CreateRaBitQComponents(allocator, dim, 1, mean);
        const auto &c = *parts.context;
        auto m = parts.preprocessors->preprocessForStorage(mean.data(), dim * 4);
        auto a = parts.preprocessors->preprocessForStorage(x.data(), dim * 4);
        auto query = parts.preprocessors->preprocessQuery(q.data(), dim * 4);
        EXPECT_EQ(c.factors(m.get()).rescale, 0);
        EXPECT_EQ(c.factors(m.get()).add, 1);
        const auto decoded = decodedReference(c, m.get(), allocator);
        double norm = 0;
        for (size_t i = 0; i < decoded.size(); ++i) {
            EXPECT_EQ(decoded[i], c.rotatedMean()[i]);
            norm += decoded[i] * decoded[i];
        }
        EXPECT_NEAR(c.storedDistance(m.get(), m.get()), 1 - norm, 1e-7);
        const double bias = double(c.factors(a.get()).add) - 1;
        ASSERT_GT(std::abs(bias), 1e-5);
        EXPECT_NEAR(c.queryDistance(a.get(), query.get()),
                    queryReference(c, a.get(), c.queryVector(query.get()), allocator), 8e-6);
    }
}

TEST(RaBitQ1, UnalignedInputAndSeparatePreparedQueriesOwnFullFp32Payloads) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    for (size_t dim : {65, 97, 768}) {
        auto parts = CreateRaBitQComponents(allocator, dim, 1);
        const auto &c = *parts.context;
        const auto x = unitVector(dim, 4), y = unitVector(dim, 9);
        std::vector<float> storage(dim + 8);
        float *unaligned = storage.data();
        while (reinterpret_cast<uintptr_t>(unaligned) % 16 != 4) ++unaligned;
        std::copy(x.begin(), x.end(), unaligned);
        auto record = parts.preprocessors->preprocessForStorage(unaligned, dim * 4);
        auto aligned_record = parts.preprocessors->preprocessForStorage(x.data(), dim * 4);
        EXPECT_EQ(std::memcmp(record.get(), aligned_record.get(), c.storageBytes()), 0);
        auto qx = parts.preprocessors->preprocessQuery(unaligned, dim * 4);
        const float before = c.queryDistance(record.get(), qx.get());
        std::copy(y.begin(), y.end(), unaligned);
        auto qy = parts.preprocessors->preprocessQuery(unaligned, dim * 4);
        std::fill(storage.begin(), storage.end(), 0);
        std::vector<float> rotated_x(c.paddedDim()), rotated_y(c.paddedDim());
        c.rotate(x.data(), rotated_x.data());
        c.rotate(y.data(), rotated_y.data());
        EXPECT_EQ(std::memcmp(c.queryVector(qx.get()), rotated_x.data(), c.paddedDim() * 4), 0);
        EXPECT_EQ(std::memcmp(c.queryVector(qy.get()), rotated_y.data(), c.paddedDim() * 4), 0);
        EXPECT_EQ(c.queryDistance(record.get(), qx.get()), before);
        EXPECT_NEAR(c.queryDistance(record.get(), qy.get()),
                    queryReference(c, record.get(), rotated_y.data(), allocator), 8e-6);
    }
}

TEST(RaBitQ1, ValidationDoesNotLeakOrAcceptOutOfContractInputs) {
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const auto baseline = allocator->getAllocationSize();
    for (size_t bad_dim : {size_t{0}, size_t{63}, size_t{65537}, std::numeric_limits<size_t>::max()})
        EXPECT_THROW(CreateRaBitQComponents(allocator, bad_dim, 1), std::invalid_argument);
    for (size_t bad_bits : {0, 4})
        EXPECT_THROW(CreateRaBitQComponents(allocator, 65, bad_bits), std::invalid_argument);
    EXPECT_THROW(CreateRaBitQComponents(nullptr, 65, 1), std::invalid_argument);
    EXPECT_THROW(CreateRaBitQComponents(allocator, 65, 1, std::vector<float>(64)), std::invalid_argument);
    EXPECT_EQ(allocator->getAllocationSize(), baseline);
    auto parts = CreateRaBitQComponents(allocator, 65, 1);
    auto x = unitVector(65, 9);
    const auto shared = allocator->getAllocationSize();
    EXPECT_THROW(parts.preprocessors->preprocessQuery(nullptr, 65 * 4), std::invalid_argument);
    EXPECT_THROW(parts.preprocessors->preprocessForStorage(x.data(), 64 * 4), std::invalid_argument);
    EXPECT_THROW(parts.preprocessors->preprocessQuery(x.data(), 66 * 4), std::invalid_argument);
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(), 2.0F}) {
        x[13] = invalid;
        EXPECT_THROW(parts.preprocessors->preprocessForStorage(x.data(), 65 * 4), std::invalid_argument);
        EXPECT_THROW(parts.preprocessors->preprocessQuery(x.data(), 65 * 4), std::invalid_argument);
        EXPECT_THROW(CreateRaBitQComponents(allocator, 65, 1, x), std::invalid_argument);
        EXPECT_EQ(allocator->getAllocationSize(), shared);
    }
}

TEST(RaBitQ1, AllocationBudgetsAndBlobLifetimesIncludeExternalPayloads) {
    for (size_t dim : {65, 97, 768}) {
        auto allocator = VecSimAllocator::newVecsimAllocator();
        const auto baseline = allocator->getAllocationSize();
        MemoryUtils::unique_blob record(nullptr, [](void *) {}), query(nullptr, [](void *) {});
        size_t retained_bytes = 0;
        {
            auto mean = unitVector(dim, 3);
            for (float &v : mean) v *= 0.2F;
            auto parts = CreateRaBitQComponents(allocator, dim, 1, mean);
            const auto &c = *parts.context;
            EXPECT_EQ(c.sharedExternalBytes(), c.paddedDim() / 2);
            EXPECT_EQ(c.sharedMeanBytes(), c.paddedDim() * sizeof(float));
            EXPECT_EQ(c.queryBytes(), sizeof(rabitq_api::QueryState) + c.paddedDim() * sizeof(float));
            EXPECT_EQ(c.queryAllocationBytes(), c.queryBytes());
            EXPECT_EQ(c.transientBlobDeleterExternalBytes(), sizeof(std::shared_ptr<VecSimAllocator>));
            EXPECT_EQ(c.encodingTemporaryBytes(), c.paddedDim() * sizeof(float) +
                c.requirements().workspace_bytes + c.requirements().workspace_alignment);
            auto x = unitVector(dim, 17);
            const auto shared = allocator->getAllocationSize();
            record = parts.preprocessors->preprocessForStorage(x.data(), dim * 4);
            EXPECT_EQ(allocator->getAllocationSize() - shared,
                      c.storageBytes() + VecSimAllocator::getAllocationOverheadSize());
            query = parts.preprocessors->preprocessQuery(x.data(), dim * 4);
            retained_bytes = c.storageBytes() + c.queryAllocationBytes() +
                             2 * VecSimAllocator::getAllocationOverheadSize();
            EXPECT_EQ(allocator->getAllocationSize() - shared, retained_bytes);
            EXPECT_EQ(reinterpret_cast<uintptr_t>(query.get()) % alignof(rabitq_api::QueryState), 0);
        }
        // QueryState borrows its own blob's FP32 payload; it survives all adapter owners.
        EXPECT_EQ(allocator->getAllocationSize() - baseline, retained_bytes);
        auto verifier = CreateRaBitQComponents(allocator, dim, 1);
        EXPECT_TRUE(std::isfinite(verifier.context->queryDistance(record.get(), query.get())));
        verifier = {};
        record.reset();
        query.reset();
        EXPECT_EQ(allocator->getAllocationSize(), baseline);
        // Batch iterators intentionally discard the std::function deleter and free
        // the allocation base directly. No hidden prefix is permitted.
        {
            auto parts = CreateRaBitQComponents(allocator, dim, 1);
            auto x = unitVector(dim, 5);
            auto blob = parts.preprocessors->preprocessQuery(x.data(), dim * 4);
            allocator->free_allocation(blob.release());
        }
        EXPECT_EQ(allocator->getAllocationSize(), baseline);
    }
}

TEST(RaBitQ1, ResearchHnswCrudAndBatchQueryKeepAllocatorOwnership) {
    constexpr size_t dim = 97;
    auto allocator = VecSimAllocator::newVecsimAllocator();
    const auto baseline = allocator->getAllocationSize();
    {
        auto parts = CreateRaBitQComponents(allocator, dim, 1);
        AbstractIndexInitParams init{.allocator = allocator, .dim = dim,
            .vecType = VecSimType_FLOAT32, .storedDataSize = parts.context->storageBytes(),
            .metric = VecSimMetric_IP, .blockSize = 32, .multi = false, .isDisk = false,
            .isQuantized = true, .logCtx = nullptr, .inputBlobSize = dim * sizeof(float)};
        HNSWParams hp{.type = VecSimType_FLOAT32, .dim = dim, .metric = VecSimMetric_IP,
            .multi = false, .blockSize = 32, .M = 8, .efConstruction = 40};
        std::unique_ptr<VecSimIndex, decltype(&VecSimIndex_Free)> index(
            new (allocator) HNSWIndex_Single<float, float>(&hp, init, parts.components(), 100),
            &VecSimIndex_Free);
        parts.release();
        std::vector<std::vector<float>> data;
        for (size_t i = 0; i < 24; ++i) {
            data.push_back(unitVector(dim, i + 10));
            EXPECT_EQ(VecSimIndex_AddVector(index.get(), data.back().data(), i), 1);
        }
        auto q = unitVector(dim, 42);
        EXPECT_EQ(VecSimIndex_DeleteVector(index.get(), 7), 1);
        EXPECT_EQ(VecSimIndex_AddVector(index.get(), data[7].data(), 7), 1);
        data[7] = q;
        EXPECT_EQ(VecSimIndex_AddVector(index.get(), data[7].data(), 7), 0);
        EXPECT_EQ(VecSimIndex_IndexSize(index.get()), 24);

        auto reference = CreateRaBitQComponents(allocator, dim, 1);
        const auto &c = *reference.context;
        std::vector<float> rotated_query(c.paddedDim());
        c.rotate(q.data(), rotated_query.data());
        std::vector<std::pair<double, size_t>> expected;
        for (size_t label = 0; label < data.size(); ++label) {
            auto record = reference.preprocessors->preprocessForStorage(data[label].data(), dim * 4);
            expected.emplace_back(queryReference(c, record.get(), rotated_query.data(), allocator), label);
        }
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(expected.front().second, 7);
        // Distinct reference scores make the exhaustive ranking check insensitive
        // to the FP32 reduction tolerance allowed for individual scores.
        for (size_t i = 0; i < 5; ++i)
            ASSERT_GT(expected[i + 1].first - expected[i].first, 2e-5);
        VecSimQueryParams params{};
        params.hnswRuntimeParams.efRuntime = data.size();
        std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)> reply(
            VecSimIndex_TopKQuery(index.get(), q.data(), 5, &params, BY_SCORE), &VecSimQueryReply_Free);
        ASSERT_NE(reply, nullptr);
        ASSERT_EQ(VecSimQueryReply_Len(reply.get()), 5);
        ASSERT_EQ(VecSimQueryReply_GetCode(reply.get()), VecSim_QueryReply_OK);
        std::vector<std::pair<size_t, double>> topk;
        {
            std::unique_ptr<VecSimQueryReply_Iterator, decltype(&VecSimQueryReply_IteratorFree)> it(
                VecSimQueryReply_GetIterator(reply.get()), &VecSimQueryReply_IteratorFree);
            for (size_t rank = 0; rank < 5; ++rank) {
                auto *item = VecSimQueryReply_IteratorNext(it.get());
                ASSERT_NE(item, nullptr);
                EXPECT_EQ(VecSimQueryResult_GetId(item), expected[rank].second);
                EXPECT_NEAR(VecSimQueryResult_GetScore(item), expected[rank].first, 8e-6);
                topk.emplace_back(VecSimQueryResult_GetId(item), VecSimQueryResult_GetScore(item));
            }
        }
        std::unique_ptr<VecSimBatchIterator, decltype(&VecSimBatchIterator_Free)> batch(
            VecSimBatchIterator_New(index.get(), q.data(), &params), &VecSimBatchIterator_Free);
        std::unique_ptr<VecSimQueryReply, decltype(&VecSimQueryReply_Free)> batch_reply(
            VecSimBatchIterator_Next(batch.get(), 5, BY_SCORE), &VecSimQueryReply_Free);
        ASSERT_NE(batch_reply, nullptr);
        ASSERT_EQ(VecSimQueryReply_Len(batch_reply.get()), 5);
        ASSERT_EQ(VecSimQueryReply_GetCode(batch_reply.get()), VecSim_QueryReply_OK);
        std::unique_ptr<VecSimQueryReply_Iterator, decltype(&VecSimQueryReply_IteratorFree)> batch_it(
            VecSimQueryReply_GetIterator(batch_reply.get()), &VecSimQueryReply_IteratorFree);
        for (size_t rank = 0; rank < topk.size(); ++rank) {
            auto *item = VecSimQueryReply_IteratorNext(batch_it.get());
            ASSERT_NE(item, nullptr);
            EXPECT_EQ(VecSimQueryResult_GetId(item), expected[rank].second);
            EXPECT_NEAR(VecSimQueryResult_GetScore(item), expected[rank].first, 8e-6);
            EXPECT_EQ(VecSimQueryResult_GetId(item), topk[rank].first);
            EXPECT_EQ(VecSimQueryResult_GetScore(item), topk[rank].second);
        }
    }
    EXPECT_EQ(allocator->getAllocationSize(), baseline);
}
