/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3.
 */
#pragma once
#include <type_traits>
#include "VecSim/quantization/turboquant.h"
#include "VecSim/quantization/quantization_runtime.h"
#include "VecSim/spaces/computer/calculator.h"

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

// The common component wrappers share this storage context. The codec remains stable when
// its fitted mean changes, so previously captured distance dispatches remain valid.
class TurboQuantStorageContextBase : public QuantizationRuntime {
protected:
    std::shared_ptr<VecSimAllocator> allocator_;
    std::shared_ptr<TurboQuant> codec_;
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
    TurboQuantStorageContextBase(const std::shared_ptr<VecSimAllocator> &allocator,
                                 std::shared_ptr<TurboQuant> codec)
        : allocator_(allocator), codec_(std::move(codec)) {}

public:
    using QueryType = float;
    const std::shared_ptr<TurboQuant> &codec() const { return codec_; }
    void encodeStorage(const void *input, void *output) const {
        validateTurboQuantInput(static_cast<const float *>(input), codec_->dim());
        vecsim_stl::vector<float> residual(codec_->dim(), allocator_);
        const auto values = static_cast<const float *>(input);
        const auto mean = codec_->biasMean();
        for (size_t i = 0; i < mean.size(); ++i)
            residual[i] = values[i] - mean[i];
        codec_->encode(residual.data(), output);
    }
    void prepareQuery(const void *input, void *output) const {
        validateTurboQuantInput(static_cast<const float *>(input), codec_->dim());
        codec_->prepareQuery(static_cast<const float *>(input), output);
        float meanIP = 0;
        const auto mean = codec_->biasMean();
        const auto values = static_cast<const float *>(input);
        for (size_t i = 0; i < mean.size(); ++i)
            meanIP += mean[i] * values[i];
        if (!std::isfinite(meanIP))
            throw std::overflow_error("TurboQuant mean IP overflow");
        std::memcpy(static_cast<unsigned char *>(output) + codec_->queryBytes(), &meanIP,
                    sizeof(meanIP));
    }
    void encodeStorageInPlace(void *input, size_t bytes) const {
        if (bytes != codec_->dim() * sizeof(float) || bytes < storageBytes())
            throw std::invalid_argument("TurboQuant expects a fitting FP32 vector buffer");
        auto result = allocator_->allocate_unique(storageBytes());
        encodeStorage(input, result.get());
        std::memcpy(input, result.get(), storageBytes());
    }
    unsigned char storageAlignment() const { return 0; }
    unsigned char queryAlignment() const { return alignof(float); }
    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const {
        return DistanceDispatch<float>::stateful(codec_.get(),
            mode == DistanceMode::StoredToStored ? stored : query);
    }
    void validateBatch(const void *rows, size_t count, size_t rowBytes,
                       VecSimMetric inputMetric) const override {
        validateTurboQuantBatch(rows, count, codec_->dim(), rowBytes,
                                inputMetric == VecSimMetric_Cosine);
    }
    void setMean(std::span<const float> mean) override {
        if (mean.size() != codec_->dim())
            throw std::invalid_argument("TurboQuant mean dimension mismatch");
        validateTurboQuantInput(mean.data(), mean.size());
        auto fitted = codec_->prepareMean(mean);
        codec_->publishMean(std::move(fitted));
    }
    std::span<const float> mean() const override { return codec_->biasMean(); }
    size_t storageBytes() const override { return codec_->storageBytes(); }
    size_t queryBytes() const override { return codec_->queryBytes() + sizeof(float); }
    QuantizationInfo info() const override {
        return {.type = codec_->bits() == 2 ? VecSimQuant_TQ2 : VecSimQuant_TQ3,
                .name = codec_->bits() == 2 ? "TQ2" : "TQ3",
                .scale = codec_->scalePolicyName(),
                .bias = codec_->biasPolicyName(),
                .vectorBytes = storageBytes(),
                .queryBytes = queryBytes(),
                .sharedBytes = codec_->sharedDataBytes(),
                .queryKernel = codec_->scoringKernelName(),
                .graphKernel = codec_->storedScoringKernelName()};
    }
#ifdef BUILD_TESTS
    unsigned encodingVersion() const override { return 6; }
    void saveState(std::ostream &output) const override { codec_->saveFitted(output); }
    void validateStoredRecord(const void *record) const override {
        float metadata[2];
        std::memcpy(metadata, static_cast<const char *>(record) + storageBytes() - sizeof(metadata),
                    sizeof(metadata));
        if (!std::isfinite(metadata[0]) || metadata[0] < 0 || !std::isfinite(metadata[1]))
            throw std::runtime_error("Invalid V6 TurboQuant record");
    }
#endif
};

template <unsigned Bits, typename DataType, VecSimMetric Metric, bool WithMean>
class TurboQuantStorageContext final : public TurboQuantStorageContextBase {
    static_assert(Bits == 2 || Bits == 3);
    static_assert(std::is_same_v<DataType, float> && Metric == VecSimMetric_IP);
    static std::shared_ptr<TurboQuant>
    createCodec(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                const QuantizationInit &init) {
        turboQuantStorageBytes(dim, Bits);
#ifdef BUILD_TESTS
        if (init.fittedState)
            return TurboQuant::loadFitted(*init.fittedState, allocator, dim, Bits);
#endif
        const auto *params = static_cast<const VecSimTurboQuantParams *>(init.parameters);
        const float *mean = params && !init.prepareForTraining ? params->mean : nullptr;
        vecsim_stl::vector<float> zero(allocator);
        if (!mean) {
            zero.resize(dim, 0.0f);
            mean = zero.data();
        }
        validateTurboQuantInput(mean, dim);
        return std::allocate_shared<TurboQuant>(
            VecsimSTLAllocator<TurboQuant>(allocator), allocator, dim, Bits,
            params ? params->seed : VECSIM_TURBOQUANT_DEFAULT_SEED, TurboQuantVariant::MSE,
            TurboQuantExecution::Dispatch, TurboQuantScalePolicy::ProjectionMatch,
            TurboQuantBiasPolicy::MeanError, std::span<const float>(mean, dim));
    }

public:
    TurboQuantStorageContext(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                             const QuantizationInit &init)
        : TurboQuantStorageContextBase(allocator, createCodec(allocator, dim, init)) {}
    // Diagnostic references can select execution while using the production wrappers.
    TurboQuantStorageContext(const std::shared_ptr<VecSimAllocator> &allocator,
                             std::shared_ptr<TurboQuant> codec)
        : TurboQuantStorageContextBase(allocator, std::move(codec)) {
        if (codec_->bits() != Bits)
            throw std::invalid_argument("TurboQuant bit width mismatch");
    }
    static size_t estimateMemory(size_t dim, const QuantizationInit &) {
        turboQuantStorageBytes(dim, Bits);
        const size_t overhead = VecSimAllocator::getAllocationOverheadSize();
        return sizeof(TurboQuantStorageContext) + sizeof(TurboQuant) +
               sizeof(VecsimSTLAllocator<TurboQuant>) + 4 * sizeof(void *) + 5 * overhead +
               (dim * dim + (size_t(1) << Bits) + 2 * dim) * sizeof(float);
    }
};
} // namespace vecsim::quantization
