/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <stdexcept>
#ifdef BUILD_TESTS
#include <ostream>
#endif

#include "VecSim/quantization/quantization_runtime.h"
#include "VecSim/spaces/computer/calculator.h"
#include "VecSim/spaces/computer/preprocessors.h"

// Scalar storage policies supply layout/code hooks. Other families can select their own
// Context through StorageType::Context without changing the component wrappers.
template <typename StorageType, QuantInput Input, VecSimMetric Metric, bool WithMean>
class ScalarStorageContext : public QuantizationRuntime {
    using Engine = ScalarQuantPreprocessor<Input, Metric, WithMean, StorageType>;
    using Calculator =
        std::conditional_t<WithMean, DistanceCalculatorWithNorm<Input, float, Metric, StorageType>,
                           DistanceCalculatorCommon<float>>;

public:
    using QueryType = typename Engine::QueryType;

    ScalarStorageContext(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                         const QuantizationInit &init)
        : dim_(dim), engine_(makeEngine(allocator, dim, init)),
          calculator_(makeCalculator(allocator, dim)),
          stored_dispatch_(calculator_.getDistanceDispatch(DistanceMode::StoredToStored)),
          query_dispatch_(calculator_.getDistanceDispatch(DistanceMode::StoredToQuery)) {}

    ScalarStorageContext(const ScalarStorageContext &) = delete;
    ScalarStorageContext &operator=(const ScalarStorageContext &) = delete;

    void encodeStorage(const void *input, void *output) const {
        engine_.encodeStorage(input, output);
    }
    void prepareQuery(const void *input, void *output) const {
        engine_.prepareQuery(input, output);
    }
    void encodeStorageInPlace(void *input, size_t inputBytes) const {
        engine_.preprocessStorageInPlace(input, inputBytes);
    }
    unsigned char storageAlignment() const { return storage_alignment_; }
    unsigned char queryAlignment() const { return query_alignment_; }
    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const {
        return mode == DistanceMode::StoredToStored ? stored_dispatch_ : query_dispatch_;
    }

    void validateBatch(const void *, size_t, size_t, VecSimMetric) const override {}

    // Call only before storing vectors and with concurrent queries excluded. Validation precedes
    // publication; once it succeeds neither the engine nor the cached calculator update can throw.
    void setMean(std::span<const float> values) override {
        if constexpr (WithMean) {
            if (values.size() != dim_) {
                throw std::invalid_argument("Mean vector size must equal dimension");
            }
            engine_.setMean(values);
            calculator_.setMeanSumSquares(values);
        } else {
            if (!values.empty()) {
                throw std::invalid_argument("Storage context does not use a mean");
            }
        }
    }
    std::span<const float> mean() const override { return engine_.getMean(); }
    size_t storageBytes() const override {
        return StorageType::template storage_bytes_count<Metric, WithMean>(dim_);
    }
    size_t queryBytes() const override {
        return dim_ * sizeof(QueryType) +
               StorageType::template query_metadata_count<Metric, WithMean>() * sizeof(float);
    }
    QuantizationInfo info() const override {
        QuantizationInfo result;
        if constexpr (requires { StorageType::quantization_type; }) {
            result.type = StorageType::quantization_type;
        }
        result.vectorBytes = storageBytes();
        result.queryBytes = queryBytes();
        result.sharedBytes = mean().size_bytes();
        return result;
    }

    // Includes owned dynamic state; the common estimator separately counts allocate_shared's
    // implementation-defined control block and the component wrappers.
    static size_t estimateMemory(size_t dim, const QuantizationInit &) {
        return sizeof(ScalarStorageContext) +
               (WithMean ? dim * sizeof(float) + VecSimAllocator::getAllocationOverheadSize() : 0);
    }
#ifdef BUILD_TESTS
    unsigned encodingVersion() const override { return 5; }
    void saveState(std::ostream &output) const override {
        const bool has_mean = WithMean;
        output.write(reinterpret_cast<const char *>(&has_mean), sizeof(has_mean));
        if constexpr (WithMean) {
            const auto values = mean();
            output.write(reinterpret_cast<const char *>(values.data()), values.size_bytes());
        }
    }
    void validateStoredRecord(const void *) const override {}
#endif

private:
    static Engine makeEngine(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                             const QuantizationInit &init) {
        if constexpr (WithMean) {
            vecsim_stl::vector<float> values(dim, 0.0f, allocator);
            if (!init.prepareForTraining) {
                const auto *mean = static_cast<const float *>(init.parameters);
                if (!mean) {
                    throw std::invalid_argument("Mean-normalized storage requires a mean");
                }
                std::copy_n(mean, dim, values.begin());
            }
            return Engine(allocator, dim, values);
        } else {
            return Engine(allocator, dim);
        }
    }
    Calculator makeCalculator(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim) {
        unsigned char asym_alignment = 0;
        auto sym = spaces::GetDistFunc<StorageType, float>(Metric, dim, &storage_alignment_);
        auto asym =
            spaces::GetDistFunc<StorageType, float, QueryType>(Metric, dim, &asym_alignment);
        storage_alignment_ = spaces::combineAlignments(storage_alignment_, asym_alignment);
        spaces::GetDistFunc<QueryType, float>(Metric, dim, &query_alignment_);
        if constexpr (WithMean) {
            float sum_squares = 0.0f;
            for (float value : engine_.getMean()) {
                sum_squares += value * value;
            }
            return Calculator(allocator, asym, sym, sum_squares);
        } else {
            return Calculator(allocator, sym, asym);
        }
    }

    const size_t dim_;
    unsigned char storage_alignment_ = 0;
    unsigned char query_alignment_ = 0;
    Engine engine_;
    Calculator calculator_;
    DistanceDispatch<float> stored_dispatch_;
    DistanceDispatch<float> query_dispatch_;
};
