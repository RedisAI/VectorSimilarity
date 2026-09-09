/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/spaces/computer/quantization_trainer.h"
#include "VecSim/spaces/computer/preprocessors.h"
#include "VecSim/spaces/computer/calculator.h"

template <typename DataType, VecSimMetric Metric>
class SQ8QuantizationTrainer final : public QuantizationTrainer<DataType> {
    static_assert(QuantInput<DataType>);

    vecsim_stl::vector<double> runningSum;
    size_t vectorCount = 0;
    const size_t threshold;
    QuantPreprocessor<DataType, Metric, true> &preprocessor;
    DistanceCalculatorWithNorm<DataType, float, Metric> &calculator;

public:
    SQ8QuantizationTrainer(std::shared_ptr<VecSimAllocator> allocator, size_t dim, size_t threshold,
                           QuantPreprocessor<DataType, Metric, true> &preprocessor,
                           DistanceCalculatorWithNorm<DataType, float, Metric> &calculator)
        : QuantizationTrainer<DataType>(allocator), runningSum(dim, 0.0, allocator),
          threshold(threshold), preprocessor(preprocessor), calculator(calculator) {
        assert(threshold > 0);
        assert(preprocessor.mean.size() == dim);
    }

    void addVector(std::span<const DataType> vector) override {
        assert(vector.size() == runningSum.size());
        for (size_t i = 0; i < vector.size(); ++i) {
            runningSum[i] += to_fp32(vector[i]);
        }
        ++vectorCount;
    }

    void removeVector(std::span<const DataType> vector) override {
        assert(vector.size() == runningSum.size());
        assert(vectorCount > 0);
        for (size_t i = 0; i < vector.size(); ++i) {
            runningSum[i] -= to_fp32(vector[i]);
        }
        --vectorCount;
    }

    bool ready() const override { return vectorCount >= threshold; }

    void finalize() noexcept override {
        assert(ready());
        float meanSumSquares = 0.0f;
        for (size_t i = 0; i < runningSum.size(); ++i) {
            const float mean = static_cast<float>(runningSum[i] / static_cast<double>(vectorCount));
            preprocessor.mean[i] = mean;
            meanSumSquares += mean * mean;
        }
        // Cached IP distance dispatches point into this context. Update its value in place;
        // replacing the calculator would invalidate those pointers.
        calculator.context_.mean_sum_squares = meanSumSquares;
    }

#ifdef BUILD_TESTS
    const auto &getRunningSum() const { return runningSum; }
    size_t getVectorCount() const { return vectorCount; }
    size_t getThreshold() const { return threshold; }
#endif
};
