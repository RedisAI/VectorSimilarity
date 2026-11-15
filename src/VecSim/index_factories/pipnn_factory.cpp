/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/index_factories/pipnn_factory.h"
#include "VecSim/algorithms/pipnn/pipnn.h"
#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/index_factories/components/components_factory.h"
#include "VecSim/index_factories/factory_utils.h"
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

namespace PiPNNFactory {
namespace {
size_t CheckedAdd(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        throw std::overflow_error("PiPNN allocation size overflow");
    }
    return a + b;
}
size_t CheckedMultiply(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) {
        throw std::overflow_error("PiPNN allocation size overflow");
    }
    return a * b;
}
BFParams ToBFParams(const PiPNNParams &params) {
    return {.type = params.type,
            .dim = params.dim,
            .metric = params.metric,
            .multi = false,
            .initialCapacity = params.initialCapacity,
            .blockSize = params.blockSize};
}
} // namespace

PiPNNParams ResolveParams(const PiPNNParams &input) {
    if (input.type != VecSimType_FLOAT32 || input.metric != VecSimMetric_L2 || input.multi ||
        input.dim == 0) {
        throw std::invalid_argument("PiPNN requires FLOAT32, L2, positive dimension, single-value");
    }
    PiPNNParams params = input;
    if (!params.blockSize) {
        params.blockSize = DEFAULT_BLOCK_SIZE;
    }
    if (!params.maxDegree) {
        params.maxDegree = 32;
    }
    if (!params.leafSize) {
        params.leafSize = 128;
    }
    if (!params.fanout) {
        params.fanout = 2;
    }
    if (!params.hashBits) {
        params.hashBits = 12;
    }
    if (!params.candidateLimit) {
        params.candidateLimit = 80;
    }
    if (params.alpha == 0.0) {
        params.alpha = 1.1;
    }
    if (!params.efRuntime) {
        params.efRuntime = 100;
    }
    if (!params.seed) {
        params.seed = 100;
    }
    constexpr size_t max_id = std::numeric_limits<idType>::max();
    if (params.maxDegree < 2 || params.maxDegree > max_id || params.leafSize < 2 ||
        params.leafSize > max_id || params.fanout < 2 || params.fanout > 32 ||
        params.hashBits > 32 || params.candidateLimit < params.maxDegree ||
        params.candidateLimit > max_id || params.efRuntime > max_id || params.blockSize > max_id ||
        params.initialCapacity > max_id || !std::isfinite(params.alpha) || params.alpha < 1.0) {
        throw std::invalid_argument("Invalid PiPNN tuning parameters");
    }

    // Check all parameter-dependent allocation products before dispatching or estimating.
    // Include allocator headers/alignment and metadata, not just the vector payload.
    const size_t overhead = CheckedAdd(VecSimAllocator::getAllocationOverheadSize(), 64);
    const size_t vector_bytes = CheckedMultiply(params.dim, sizeof(float));
    CheckedAdd(CheckedMultiply(params.blockSize, vector_bytes), overhead);
    CheckedAdd(CheckedMultiply(params.blockSize, sizeof(labelType)), overhead);
    CheckedAdd(CheckedMultiply(params.maxDegree, sizeof(idType)), overhead);
    CheckedAdd(CheckedMultiply(params.candidateLimit, sizeof(std::pair<float, idType>)), overhead);
    CheckedAdd(CheckedMultiply(params.efRuntime, sizeof(std::pair<float, idType>)), overhead);
    CheckedAdd(CheckedMultiply(CheckedMultiply(params.fanout, params.dim), sizeof(float)),
               overhead);
    CheckedAdd(CheckedMultiply(CheckedMultiply(params.hashBits, params.dim), sizeof(float)),
               overhead);
    if (params.initialCapacity) {
        const size_t rounded = CheckedAdd(params.initialCapacity, params.blockSize - 1) /
                               params.blockSize * params.blockSize;
        if (rounded > max_id) {
            throw std::invalid_argument("PiPNN capacity exceeds internal ID range");
        }
        CheckedMultiply(rounded, vector_bytes);
        CheckedMultiply(CheckedMultiply(rounded, params.maxDegree), sizeof(idType));
    }
    return params;
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    if (!params || params->algo != VecSimAlgo_PIPNN) {
        throw std::invalid_argument("PiPNN factory requires PiPNN parameters");
    }
    const auto resolved = ResolveParams(params->algoParams.pipnnParams);
    const auto init = VecSimFactory::NewAbstractInitParams(&resolved, params->logCtx, false);
    init.allocator->setThrowOnAllocationFailure();
    // Reserve the index allocation before constructing components. Its guard also releases
    // the storage if any part of the index constructor fails.
    auto storage = init.allocator->allocate_unique(sizeof(PiPNNIndex));
    unsigned char alignment = 0;
    auto dist_func = spaces::GetDistFunc<float, float>(resolved.metric, resolved.dim, &alignment);
    std::unique_ptr<IndexCalculatorInterface<float>> calculator(
        new (init.allocator) DistanceCalculatorCommon<float>(init.allocator, dist_func));
    // PiPNN only accepts L2, so this container does not need normalization preprocessors.
    std::unique_ptr<PreprocessorsContainerAbstract> preprocessors(
        new (init.allocator) PreprocessorsContainerAbstract(init.allocator, alignment));
    const IndexComponents<float, float> components{calculator.release(), preprocessors.release()};
    // VecSimIndexAbstract adopts both components, including cleanup if its storage allocation
    // fails. A completed base destructor owns them if a later constructor fails.
    auto *index = ::new (storage.get()) PiPNNIndex(&resolved, init, components);
    storage.release();
    return index;
}

size_t EstimateInitialSize(const PiPNNParams *params) {
    if (!params) {
        throw std::invalid_argument("Missing PiPNN parameters");
    }
    const auto resolved = ResolveParams(*params);
    const auto bf = ToBFParams(resolved);
    const size_t base = BruteForceFactory::EstimateInitialSize(&bf);
    return CheckedAdd(base - sizeof(BruteForceIndex_Single<float, float>), sizeof(PiPNNIndex));
}

size_t EstimateElementSize(const PiPNNParams *params) {
    if (!params) {
        throw std::invalid_argument("Missing PiPNN parameters");
    }
    const auto resolved = ResolveParams(*params);
    const auto bf = ToBFParams(resolved);
    // The built graph stores one degree and a fixed-width row of IDs per node.
    size_t size = BruteForceFactory::EstimateElementSize(&bf);
    size = CheckedAdd(size, sizeof(size_t));
    return CheckedAdd(size, CheckedMultiply(resolved.maxDegree, sizeof(idType)));
}
} // namespace PiPNNFactory
