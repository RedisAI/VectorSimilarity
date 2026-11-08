/* Copyright (c) 2006-Present, Redis Ltd. All rights reserved.
 * Licensed under your choice of RSALv2, SSPLv1, or AGPLv3. */
#pragma once

#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/computer/preprocessor_container.h"
#include "VecSim/utils/vecsim_stl.h"
#include <rabitqlib/core/api.hpp>
#include <rabitqlib/core/rotator.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace vecsim::experimental {
namespace rabitq_api = rabitqlib::quant::v1;

namespace rabitq_research_detail {
// Keep the allocation base pointer: generic batch iterators release query blobs
// and later free the base directly. Shared capture also makes ordinary blobs safe
// after their preprocessor dies. Its std::function allocation is explicitly reported.
inline MemoryUtils::unique_blob allocateBlob(const std::shared_ptr<VecSimAllocator> &allocator,
                                             size_t bytes) {
    void *allocation = allocator->allocate(bytes);
    if (!allocation)
        throw std::bad_alloc();
    try {
        return MemoryUtils::unique_blob(allocation,
            [allocator](void *data) { allocator->free_allocation(data); });
    } catch (...) {
        allocator->free_allocation(allocation);
        throw;
    }
}
inline size_t checkedDimension(size_t dim, size_t bits) {
    if (dim < 64 || dim > rabitqlib::rotator_impl::FhtKacRotator::kMaxDim ||
        (bits != 1 && bits != 2 && bits != 3))
        throw std::invalid_argument("RaBitQ research requires dimension 64..65536 and bits 1, 2 or 3");
    return (dim + 31) / 32 * 32;
}
inline void validateInput(const float *input, size_t dim) {
    if (!input)
        throw std::invalid_argument("RaBitQ research input is null");
    double norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        if (!std::isfinite(input[i]))
            throw std::invalid_argument("RaBitQ research input must be finite");
        norm += double(input[i]) * input[i];
    }
    // This prototype accepts externally normalized vectors (and zero), and a
    // frozen mean of such vectors. Check before HNSW can mutate its graph.
    if (norm > 1.001)
        throw std::invalid_argument("RaBitQ research requires normalized FP32 vectors or zero");
}
} // namespace rabitq_research_detail

// RAM-only research representation, with no public type or serializer contract.
// Record: sign bytes | extra bytes | add,rescale,error (three unaligned FP32s).
// At one bit the extra plane is absent, including in the official API views.
// The official v1 exact encoder/compute and seeded official FHT/Kac are used.
class RaBitQContext final {
    size_t dim_;
    rabitq_api::Parameters parameters_;
    rabitq_api::Requirements requirements_;
    rabitqlib::rotator_impl::FhtKacRotator rotator_;
    vecsim_stl::vector<float> rotated_mean_;

public:
    RaBitQContext(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim, size_t bits,
                  std::span<const float> mean = {}, uint32_t seed = 42)
        : dim_(dim), parameters_{rabitq_research_detail::checkedDimension(dim, bits), bits,
                                rabitqlib::METRIC_IP},
          requirements_(rabitq_api::requirements(parameters_, rabitq_api::CodeLayout::SplitPacked)),
          rotator_(dim, parameters_.dim, seed), rotated_mean_(allocator) {
        if (!allocator || (!mean.empty() && mean.size() != dim))
            throw std::invalid_argument("RaBitQ research needs an allocator and a dimension-sized mean");
        if (!mean.empty()) {
            rabitq_research_detail::validateInput(mean.data(), dim);
            rotated_mean_.resize(paddedDim());
            rotate(mean.data(), rotated_mean_.data());
        }
    }
    size_t dim() const { return dim_; }
    size_t paddedDim() const { return parameters_.dim; }
    size_t bits() const { return parameters_.total_bits; }
    const rabitq_api::Parameters &parameters() const { return parameters_; }
    const rabitq_api::Requirements &requirements() const { return requirements_; }
    size_t storageBytes() const { return requirements_.code_bytes + requirements_.extra_bytes + 12; }
    static constexpr size_t queryVectorOffset() { return sizeof(rabitq_api::QueryState); }
    size_t queryBytes() const { return queryVectorOffset() + paddedDim() * sizeof(float); }
    size_t queryAllocationBytes() const {
        return queryBytes();
    }
    // libstdc++ stores the nontrivial shared_ptr deleter capture on the heap.
    // Requested payload only; the C allocator bookkeeping is not observable here.
    static constexpr size_t transientBlobDeleterExternalBytes() {
        return sizeof(std::shared_ptr<VecSimAllocator>);
    }
    size_t encodingWorkspaceBytes() const { return requirements_.workspace_bytes; }
    size_t encodingTemporaryBytes() const {
        // Float rotation buffer plus aligned exact-encoder workspace, excluding
        // allocator headers and the output record (tracked by the allocator).
        return paddedDim() * sizeof(float) + requirements_.workspace_bytes +
               requirements_.workspace_alignment;
    }
    // The context/control block, rotator object, transformed mean (capacity),
    // components and transient blobs use VecSimAllocator. Only the official
    // rotator's private std::vector sign payload is allocated externally.
    // This is its requested payload, excluding the system malloc bookkeeping.
    size_t sharedExternalBytes() const { return rotator_.dump_bytes(); }
    size_t sharedMeanBytes() const { return rotated_mean_.capacity() * sizeof(float); }
    std::span<const float> rotatedMean() const {
        return {rotated_mean_.data(), rotated_mean_.size()};
    }
    void rotate(const float *input, float *output) const { rotator_.rotate(input, output); }
    const float *queryVector(const void *query) const {
        return reinterpret_cast<const float *>(static_cast<const unsigned char *>(query) +
                                               queryVectorOffset());
    }
    const rabitq_api::QueryState &queryState(const void *query) const {
        return *static_cast<const rabitq_api::QueryState *>(query);
    }
    rabitq_api::CodeView codeView(const void *record) const {
        return {rabitq_api::CodeLayout::SplitPacked,
                {record, requirements_.code_bytes},
                {requirements_.extra_bytes == 0 ? nullptr :
                     static_cast<const unsigned char *>(record) + requirements_.code_bytes,
                 requirements_.extra_bytes}};
    }
    rabitq_api::Factors factors(const void *record) const {
        rabitq_api::Factors result;
        const auto *metadata = static_cast<const unsigned char *>(record) + storageBytes() - 12;
        std::memcpy(&result.add, metadata, 4);
        std::memcpy(&result.rescale, metadata + 4, 4);
        std::memcpy(&result.error, metadata + 8, 4);
        return result;
    }
    // Coordinates are decoded directly from the documented SplitPacked format.
    // Sign words reverse coordinate order within a full64/final32 word; extras
    // use sequential1bit, full64 transposed2bit, or contiguous final32 tail.
    unsigned logicalCode(const void *record, size_t i) const {
        const auto *bytes = static_cast<const unsigned char *>(record);
        const size_t start = i / 64 * 64;
        const size_t width = std::min(size_t{64}, paddedDim() - start);
        const size_t bit = width - 1 - (i - start);
        const unsigned sign = (bytes[start / 8 + bit / 8] >> (bit % 8)) & 1;
        if (bits() == 1)
            return sign;
        const auto *extra = bytes + requirements_.code_bytes;
        unsigned magnitude;
        if (bits() == 2) {
            magnitude = (extra[i / 8] >> (i % 8)) & 1;
        } else if (i < paddedDim() / 64 * 64) {
            magnitude = (extra[(i / 64) * 16 + i % 16] >> ((i % 64) / 16 * 2)) & 3;
        } else {
            const size_t offset = (i - start) * 2;
            magnitude = (extra[start / 4 + offset / 8] >> (offset % 8)) & 3;
        }
        return (sign << (bits() - 1)) | magnitude;
    }
    void decode(const void *record, std::span<float> output) const {
        if (output.size() != paddedDim())
            throw std::invalid_argument("RaBitQ decode output has wrong dimension");
        const double scale = factors(record).rescale;
        const double midpoint = double((1U << bits()) - 1) / 2;
        for (size_t i = 0; i < paddedDim(); ++i)
            output[i] = static_cast<float>((rotated_mean_.empty() ? 0 : rotated_mean_[i]) -
                                          scale * (double(logicalCode(record, i)) - midpoint));
    }
    float storedDistance(const void *a, const void *b) const {
        const auto fa = factors(a), fb = factors(b);
        const double scale_a = fa.rescale, scale_b = fb.rescale;
        const double midpoint = double((1U << bits()) - 1) / 2;
        double ip = 0;
        for (size_t i = 0; i < paddedDim(); ++i) {
            const double mean = rotated_mean_.empty() ? 0 : rotated_mean_[i];
            const double wa = mean - scale_a * (double(logicalCode(a, i)) - midpoint);
            const double wb = mean - scale_b * (double(logicalCode(b, i)) - midpoint);
            ip += wa * wb;
        }
        // Symmetric extension of the official query estimator. The candidate
        // bias add-1 is kept on both sides; this graph metric is adapter-defined.
        return static_cast<float>(1 - ip + (double(fa.add) - 1) + (double(fb.add) - 1));
    }
    float queryDistance(const void *record, const void *query) const {
        // The official one-bit estimator already selects the native NEON sign
        // mask dot on ARM, with a portable scalar fallback. DirectBits only
        // selects the two-bit extra-plane kernel; it is unnecessary at one bit.
        // Keep the full FP32 query and the official correction arithmetic.
        return rabitq_api::compute(queryState(query), codeView(record), factors(record));
    }
    void encodeRotated(const float *input, void *record,
                               rabitq_api::Buffer workspace) const {
        auto *extra = requirements_.extra_bytes == 0 ? nullptr :
            static_cast<unsigned char *>(record) + requirements_.code_bytes;
        // Preserve the official centroid and all three returned factors. In IP
        // mode add can differ from one for a nonzero centroid.
        const auto f = rabitq_api::encode(parameters_, input,
            rotated_mean_.empty() ? nullptr : rotated_mean_.data(),
            {rabitq_api::CodeLayout::SplitPacked, {record, requirements_.code_bytes},
             {extra, requirements_.extra_bytes}}, workspace);
        auto *metadata = static_cast<unsigned char *>(record) + storageBytes() - 12;
        std::memcpy(metadata, &f.add, 4);
        std::memcpy(metadata + 4, &f.rescale, 4);
        std::memcpy(metadata + 8, &f.error, 4);
    }
};
static_assert(std::is_trivially_destructible_v<rabitq_api::QueryState>);
static_assert(sizeof(rabitq_api::QueryState) % alignof(float) == 0);

class RaBitQPreprocessors final : public PreprocessorsContainerAbstract {
    std::shared_ptr<const RaBitQContext> context_;
    void validate(const void *input, size_t bytes) const {
        if (bytes != context_->dim() * sizeof(float))
            throw std::invalid_argument("RaBitQ research expects one FP32 vector");
        rabitq_research_detail::validateInput(static_cast<const float *>(input), context_->dim());
    }
public:
    RaBitQPreprocessors(std::shared_ptr<VecSimAllocator> allocator,
                       std::shared_ptr<const RaBitQContext> context)
        : PreprocessorsContainerAbstract(allocator, alignof(rabitq_api::QueryState), 0),
          context_(std::move(context)) {}
    MemoryUtils::unique_blob preprocessForStorage(const void *input, size_t bytes) const override {
        validate(input, bytes);
        auto record = rabitq_research_detail::allocateBlob(allocator, context_->storageBytes());
        vecsim_stl::vector<float> residual(context_->paddedDim(), allocator);
        context_->rotate(static_cast<const float *>(input), residual.data());
        const auto &r = context_->requirements();
        auto scratch = allocator->allocate_aligned_unique(r.workspace_bytes, r.workspace_alignment);
        context_->encodeRotated(residual.data(), record.get(), {scratch.get(), r.workspace_bytes});
        return record;
    }
    MemoryUtils::unique_blob preprocessQuery(const void *input, size_t bytes,
                                             bool = false) const override {
        validate(input, bytes);
        auto blob = rabitq_research_detail::allocateBlob(allocator, context_->queryBytes());
        auto *query = const_cast<float *>(context_->queryVector(blob.get()));
        context_->rotate(static_cast<const float *>(input), query);
        const auto mean = context_->rotatedMean();
        ::new (blob.get()) rabitq_api::QueryState(rabitq_api::prepare_query(
            context_->parameters(), query, mean.empty() ? nullptr : mean.data()));
        return blob;
    }
    void preprocessStorageInPlace(void *input, size_t bytes) const override {
        if (bytes < context_->storageBytes())
            throw std::invalid_argument("RaBitQ storage does not fit the input buffer");
        auto record = preprocessForStorage(input, bytes);
        std::memcpy(input, record.get(), context_->storageBytes());
    }
};

class RaBitQCalculator final : public IndexCalculatorInterface<float> {
    std::shared_ptr<const RaBitQContext> context_;
    static float query(const void *self, const void *a, const void *b, size_t) {
        return static_cast<const RaBitQCalculator *>(self)->context_->queryDistance(a, b);
    }
    static float stored(const void *self, const void *a, const void *b, size_t) {
        return static_cast<const RaBitQCalculator *>(self)->context_->storedDistance(a, b);
    }
public:
    RaBitQCalculator(std::shared_ptr<VecSimAllocator> allocator,
                     std::shared_ptr<const RaBitQContext> context)
        : IndexCalculatorInterface<float>(allocator), context_(std::move(context)) {}
    float calcDistance(const void *a, const void *b, size_t) const override {
        return context_->storedDistance(a, b);
    }
    float calcDistanceForQuery(const void *a, const void *b, size_t) const override {
        return context_->queryDistance(a, b);
    }
    DistanceDispatch<float> getDistanceDispatch(DistanceMode mode) const override {
        return DistanceDispatch<float>::stateful(
            this, mode == DistanceMode::StoredToStored ? stored : query);
    }
};

struct RaBitQComponents {
    std::shared_ptr<const RaBitQContext> context;
    std::unique_ptr<RaBitQCalculator> calculator;
    std::unique_ptr<RaBitQPreprocessors> preprocessors;
    IndexComponents<float, float> components() const {
        return {calculator.get(), preprocessors.get()};
    }
    void release() {
        calculator.release();
        preprocessors.release();
    }
};
inline RaBitQComponents CreateRaBitQComponents(const std::shared_ptr<VecSimAllocator> &allocator,
                                               size_t dim, size_t bits,
                                               std::span<const float> mean = {}, uint32_t seed = 42) {
    if (!allocator)
        throw std::invalid_argument("RaBitQ research allocator is null");
    auto context = std::allocate_shared<RaBitQContext>(VecsimSTLAllocator<RaBitQContext>(allocator),
                                                     allocator, dim, bits, mean, seed);
    return {context,
            std::unique_ptr<RaBitQCalculator>(new (allocator) RaBitQCalculator(allocator, context)),
            std::unique_ptr<RaBitQPreprocessors>(new (allocator) RaBitQPreprocessors(allocator, context))};
}
} // namespace vecsim::experimental
