/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3.
 */
#pragma once
#include "VecSim/quantization/turboquant.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/vec_sim_index.h"

namespace vecsim::quantization {
inline bool isTurboQuant(VecSimQuantType type) {
    return type == VecSimQuant_TQ2 || type == VecSimQuant_TQ3;
}
inline unsigned turboQuantBits(VecSimQuantType type) {
    assert(isTurboQuant(type));
    return type == VecSimQuant_TQ2 ? 2 : 3;
}
inline size_t turboQuantStorageBytes(size_t dim, unsigned bits) {
    constexpr size_t limit = std::numeric_limits<size_t>::max() / (4 * sizeof(double));
    if (dim < 2 || dim > limit / dim || (bits != 2 && bits != 3))
        throw std::invalid_argument("TurboQuant requires a safe dimension >= 2 and 2/3 bits");
    return (dim * bits + 7) / 8 + 2 * sizeof(float);
}
// Reject unsafe rows before HNSW can mutate its graph. For ||x||,||mean|| <= B,
// the absolute score terms fit FP32 with margin, including the mean correction.
inline void validateTurboQuantInput(const float *input, size_t dim, bool require_nonzero = false) {
    if (!input) throw std::invalid_argument("TurboQuant input is null");
    const double bound = std::sqrt(double(std::numeric_limits<float>::max())) / (16.0 * dim);
    double squared = 0;
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(input[i]))
            throw std::invalid_argument("TurboQuant input must be finite");
        squared += double(input[i]) * input[i];
    }
    if (squared > bound * bound)
        throw std::overflow_error("TurboQuant input exceeds the safe norm bound");
    if (require_nonzero && !(squared > 0))
        throw std::invalid_argument("TurboQuant cosine input must have positive norm");
}
// Validate a complete replacement before deleting its previous label. This is validation
// only: allocation failures during an otherwise valid update retain the existing API's
// nontransactional failure contract.
inline void validateTurboQuantBatch(const void *rows, size_t count, size_t dim, size_t row_bytes,
                                    bool require_nonzero = false) {
    if (!count) return;
    if (!rows) throw std::invalid_argument("TurboQuant replacement is null");
    if (!dim || dim > std::numeric_limits<size_t>::max() / sizeof(float) ||
        row_bytes != dim * sizeof(float) ||
        count > size_t(std::numeric_limits<std::ptrdiff_t>::max()) / row_bytes ||
        reinterpret_cast<uintptr_t>(rows) > UINTPTR_MAX - count * row_bytes)
        throw std::length_error("TurboQuant replacement byte count overflows");
    const auto *bytes = static_cast<const unsigned char *>(rows);
    for (size_t i = 0; i < count; ++i)
        validateTurboQuantInput(reinterpret_cast<const float *>(bytes + i * row_bytes), dim,
                                require_nonzero);
}

class TurboQuantPreprocessors final : public PreprocessorsContainerAbstract {
    std::shared_ptr<TurboQuant> codec_;
    void check(const void *input, size_t bytes) const {
        if (bytes != codec_->dim() * sizeof(float))
            throw std::invalid_argument("TurboQuant expects one FP32 vector");
        validateTurboQuantInput(static_cast<const float *>(input), codec_->dim());
    }
public:
    TurboQuantPreprocessors(std::shared_ptr<VecSimAllocator> allocator,
                            std::shared_ptr<TurboQuant> codec)
        : PreprocessorsContainerAbstract(allocator, alignof(float), 0), codec_(std::move(codec)) {}
    const std::shared_ptr<TurboQuant> &codec() const { return codec_; }
    size_t queryBytes() const { return codec_->queryBytes() + sizeof(float); }
    MemoryUtils::unique_blob preprocessForStorage(const void *input, size_t bytes) const override {
        check(input, bytes);
        auto result = wrapAllocated(allocator->allocate(codec_->storageBytes()));
        vecsim_stl::vector<float> residual(codec_->dim(), allocator);
        const auto mean = codec_->biasMean();
        const auto *values = static_cast<const float *>(input);
        for (size_t i = 0; i < mean.size(); ++i) residual[i] = values[i] - mean[i];
        codec_->encode(residual.data(), result.get());
        return result;
    }
    MemoryUtils::unique_blob preprocessQuery(const void *input, size_t bytes,
                                             bool = false) const override {
        check(input, bytes);
        auto result = wrapAllocated(allocator->allocate(queryBytes()));
        codec_->prepareQuery(static_cast<const float *>(input), result.get());
        float meanIP = 0;
        const auto mean = codec_->biasMean();
        const auto *values = static_cast<const float *>(input);
        for (size_t i = 0; i < mean.size(); ++i) meanIP += mean[i] * values[i];
        if (!std::isfinite(meanIP)) throw std::overflow_error("TurboQuant mean IP overflow");
        std::memcpy(static_cast<unsigned char *>(result.get()) + codec_->queryBytes(),
                    &meanIP, sizeof(meanIP));
        return result;
    }
    void preprocessStorageInPlace(void *input, size_t bytes) const override {
        check(input, bytes);
        if (bytes < codec_->storageBytes())
            throw std::invalid_argument("TurboQuant storage does not fit input buffer");
        auto result = preprocessForStorage(input, bytes);
        std::memcpy(input, result.get(), codec_->storageBytes());
    }
};

class TurboQuantCalculator final : public IndexCalculatorInterface<float> {
    std::shared_ptr<const TurboQuant> codec_;
    static float stored(const void *context, const void *a, const void *b, size_t) {
        return static_cast<const TurboQuant *>(context)->storedDistance(a, b);
    }
    static float query(const void *context, const void *stored, const void *query, size_t) {
        const auto *codec = static_cast<const TurboQuant *>(context);
        float meanIP;
        std::memcpy(&meanIP, static_cast<const unsigned char *>(query) + codec->queryBytes(),
                    sizeof(meanIP));
        const float score = codec->distance(stored, query) - meanIP;
        if (!std::isfinite(score)) throw std::overflow_error("TurboQuant query score overflow");
        return score;
    }
public:
    TurboQuantCalculator(std::shared_ptr<VecSimAllocator> allocator,
                         std::shared_ptr<const TurboQuant> codec)
        : IndexCalculatorInterface<float>(allocator), codec_(std::move(codec)) {}
    float calcDistance(const void *a, const void *b, size_t dim) const override {
        return stored(codec_.get(), a, b, dim);
    }
    float calcDistanceForQuery(const void *a, const void *b, size_t dim) const override {
        return query(codec_.get(), a, b, dim);
    }
    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const override {
        return DistanceDispatch<float>::stateful(codec_.get(),
            mode == DistanceMode::StoredToStored ? stored : query);
    }
};

inline IndexComponents<float, float>
createTurboQuantComponents(const std::shared_ptr<VecSimAllocator> &allocator,
                           std::shared_ptr<TurboQuant> codec) {
    auto calculator = std::unique_ptr<TurboQuantCalculator>(
        new (allocator) TurboQuantCalculator(allocator, codec));
    auto preprocessors = std::unique_ptr<TurboQuantPreprocessors>(
        new (allocator) TurboQuantPreprocessors(allocator, std::move(codec)));
    return {calculator.release(), preprocessors.release()};
}
inline IndexComponents<float, float>
createTurboQuantComponents(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                           unsigned bits, const VecSimTurboQuantParams *params) {
    turboQuantStorageBytes(dim, bits);
    vecsim_stl::vector<float> zero(allocator);
    const float *mean = params ? params->mean : nullptr;
    if (!mean) { zero.resize(dim, 0.0f); mean = zero.data(); }
    validateTurboQuantInput(mean, dim);
    auto codec = std::allocate_shared<TurboQuant>(VecsimSTLAllocator<TurboQuant>(allocator),
        allocator, dim, bits, params ? params->seed : VECSIM_TURBOQUANT_DEFAULT_SEED,
        TurboQuantVariant::MSE, TurboQuantExecution::Dispatch,
        TurboQuantScalePolicy::ProjectionMatch, TurboQuantBiasPolicy::MeanError,
        std::span<const float>(mean, dim));
    return createTurboQuantComponents(allocator, std::move(codec));
}
// Includes every permanent allocation. Shared-control-block bookkeeping is implementation
// dependent, so reserve four pointers plus its allocator in addition to the codec object.
inline size_t estimateTurboQuantComponents(size_t dim, unsigned bits) {
    turboQuantStorageBytes(dim, bits);
    const size_t overhead = VecSimAllocator::getAllocationOverheadSize();
    return sizeof(TurboQuantCalculator) + sizeof(TurboQuantPreprocessors) + sizeof(TurboQuant) +
        sizeof(VecsimSTLAllocator<TurboQuant>) + 4 * sizeof(void *) + 7 * overhead +
        (dim * dim + (size_t(1) << bits) + 2 * dim) * sizeof(float);
}
} // namespace vecsim::quantization
