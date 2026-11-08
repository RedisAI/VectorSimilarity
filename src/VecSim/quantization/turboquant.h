/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/quantization/turboquant_kernels.h"
#include "VecSim/utils/vecsim_stl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#ifdef BUILD_TESTS
#include <istream>
#include <ostream>
#endif

namespace vecsim::quantization {

enum class TurboQuantVariant { InnerProduct, MSE };
enum class TurboQuantExecution { Scalar, Dispatch };
// Research-only encoding choices. The existing MSE scalar becomes a reconstruction
// scale; packed codes, transforms, centroids and distance arithmetic are unchanged.
enum class TurboQuantScalePolicy { Baseline, LeastSquares, NormMatch, ProjectionMatch };
enum class TurboQuantBiasPolicy { None, MeanError };

// Dense TurboQuant estimator for 1 - inner product. Transforms remain immutable.
// Per-vector buffers may be unaligned; fitted mean publication requires reader exclusion.
class TurboQuant {
    std::shared_ptr<VecSimAllocator> allocator_;
    size_t dim_;
    unsigned bits_;
    uint64_t seed_;
    TurboQuantVariant variant_;
    TurboQuantExecution execution_;
    TurboQuantScalePolicy scale_policy_;
    TurboQuantBiasPolicy bias_policy_;
    turboquant_detail::PackedDot packed_dot_ = nullptr;
    turboquant_detail::RowDot row_dot_ = nullptr;
    turboquant_detail::PackedPairDot stored_pair_dot_ = nullptr;
    unsigned code_bits_;
    size_t code_bytes_;
    size_t sign_bytes_;
    vecsim_stl::vector<float> rotation_;
    vecsim_stl::vector<float> residual_matrix_;
    vecsim_stl::vector<float> centroids_;
    // Bound once during construction; no encoding or scoring path changes these vectors.
    vecsim_stl::vector<float> bias_mean_;
    vecsim_stl::vector<float> bias_rotated_mean_;
    float bias_mean_norm_squared_ = 0;

    static size_t validatedDim(const std::shared_ptr<VecSimAllocator> &allocator, size_t dim,
                               unsigned bits, TurboQuantVariant variant,
                               TurboQuantExecution execution, TurboQuantScalePolicy scale_policy,
                               TurboQuantBiasPolicy bias_policy, std::span<const float> mean) {
        if (!allocator || dim < 2 || bits < 2 || bits > 4 ||
            (variant != TurboQuantVariant::InnerProduct && variant != TurboQuantVariant::MSE) ||
            (execution != TurboQuantExecution::Scalar &&
             execution != TurboQuantExecution::Dispatch)) {
            throw std::invalid_argument("TurboQuant requires an allocator, dim >= 2, "
                                        "bits 2..4 and valid variant/execution");
        }
        if ((scale_policy != TurboQuantScalePolicy::Baseline &&
             scale_policy != TurboQuantScalePolicy::LeastSquares &&
             scale_policy != TurboQuantScalePolicy::NormMatch &&
             scale_policy != TurboQuantScalePolicy::ProjectionMatch) ||
            (variant != TurboQuantVariant::MSE &&
             scale_policy != TurboQuantScalePolicy::Baseline)) {
            throw std::invalid_argument("TurboQuant scaled encoding requires a valid MSE policy");
        }
        if ((bias_policy != TurboQuantBiasPolicy::None &&
             bias_policy != TurboQuantBiasPolicy::MeanError) ||
            (bias_policy == TurboQuantBiasPolicy::None && !mean.empty()) ||
            (bias_policy == TurboQuantBiasPolicy::MeanError &&
             (variant != TurboQuantVariant::MSE || (bits != 2 && bits != 3) || mean.size() != dim))) {
            throw std::invalid_argument(
                "TurboQuant MeanError requires MSE2/MSE3 and a dim mean; None requires no mean");
        }
        for (float value : mean) {
            if (!std::isfinite(value))
                throw std::invalid_argument("TurboQuant bias mean must be finite");
        }
        // Bound both permanent matrices and the double-precision QR workspace before multiplying.
        constexpr size_t limit = std::numeric_limits<size_t>::max() / (4 * sizeof(double));
        if (dim > limit / dim) {
            throw std::length_error("TurboQuant dimension overflows matrix allocation");
        }
        return dim;
    }

    static uint64_t splitSeed(uint64_t seed) {
        seed += UINT64_C(0x9e3779b97f4a7c15);
        seed = (seed ^ (seed >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        seed = (seed ^ (seed >> 27)) * UINT64_C(0x94d049bb133111eb);
        return seed ^ (seed >> 31);
    }

    void initializeTransforms(uint64_t seed) {
        std::mt19937_64 rotation_rng(splitSeed(seed));
        std::mt19937_64 residual_rng(splitSeed(seed ^ UINT64_C(0xd1b54a32d192ed03)));
        std::normal_distribution<double> gaussian;
        vecsim_stl::vector<double> basis(dim_ * dim_, allocator_);
        // Row-wise Gaussian QR with positive diagonal convention gives Haar orthogonal Pi.
        // Reorthogonalization and double workspace keep the initialization stable at large d.
        for (size_t row = 0; row < dim_; ++row) {
            double *current = basis.data() + row * dim_;
            for (size_t col = 0; col < dim_; ++col) {
                current[col] = gaussian(rotation_rng);
            }
            for (unsigned pass = 0; pass < 2; ++pass) {
                for (size_t previous = 0; previous < row; ++previous) {
                    const double *other = basis.data() + previous * dim_;
                    double dot = 0;
                    for (size_t col = 0; col < dim_; ++col) {
                        dot += current[col] * other[col];
                    }
                    for (size_t col = 0; col < dim_; ++col) {
                        current[col] -= dot * other[col];
                    }
                }
            }
            double norm = 0;
            for (size_t col = 0; col < dim_; ++col) {
                norm += current[col] * current[col];
            }
            norm = std::sqrt(norm);
            if (!(norm > 0) || !std::isfinite(norm)) {
                throw std::runtime_error("TurboQuant Gaussian QR is degenerate");
            }
            for (size_t col = 0; col < dim_; ++col) {
                current[col] /= norm;
                rotation_[row * dim_ + col] = static_cast<float>(current[col]);
            }
        }
        gaussian.reset();
        for (float &value : residual_matrix_) {
            value = static_cast<float>(gaussian(residual_rng));
        }
    }

    void initializeBiasMean(std::span<const float> mean) {
        if (!hasBias())
            return;
        bias_mean_.assign(mean.begin(), mean.end());
        double norm_squared = 0;
        for (float value : bias_mean_)
            norm_squared += double(value) * value;
        bias_mean_norm_squared_ = checkedFloat(static_cast<float>(norm_squared));
        bias_rotated_mean_.resize(dim_);
        for (size_t row = 0; row < dim_; ++row) {
            double dot = 0;
            for (size_t col = 0; col < dim_; ++col)
                dot += double(rotation_[row * dim_ + col]) * bias_mean_[col];
            bias_rotated_mean_[row] = checkedFloat(static_cast<float>(dot));
        }
    }

    void initializeCentroids() {
        // The exact finite-d sphere marginal is (1-z*z)^((d-3)/2). With z=sin(theta),
        // integration weight becomes cos(theta)^(d-2), including the singular d=2 case.
        // Midpoint quadrature approximates its Lloyd-Max solution; it is not a Gaussian codebook.
        constexpr size_t samples = 32768;
        vecsim_stl::vector<double> coordinates(samples, allocator_);
        vecsim_stl::vector<double> weights(samples, allocator_);
        const size_t levels = centroids_.size();
        vecsim_stl::vector<double> centers(levels, allocator_);
        vecsim_stl::vector<double> masses(levels, allocator_);
        vecsim_stl::vector<double> sums(levels, allocator_);
        for (size_t sample = 0; sample < samples; ++sample) {
            const double theta = std::numbers::pi * ((sample + 0.5) / samples - 0.5);
            coordinates[sample] = std::sin(theta);
            weights[sample] = std::pow(std::cos(theta), static_cast<double>(dim_ - 2));
        }
        for (size_t level = 0; level < levels; ++level) {
            centers[level] = (2.0 * level + 1.0 - levels) / levels *
                             std::min(1.0, 2.5 / std::sqrt(static_cast<double>(dim_)));
        }
        bool converged = false;
        for (unsigned iteration = 0; iteration < 1024; ++iteration) {
            std::fill(masses.begin(), masses.end(), 0);
            std::fill(sums.begin(), sums.end(), 0);
            size_t level = 0;
            for (size_t sample = 0; sample < samples; ++sample) {
                while (level + 1 < levels &&
                       coordinates[sample] > (centers[level] + centers[level + 1]) * 0.5) {
                    ++level;
                }
                masses[level] += weights[sample];
                sums[level] += weights[sample] * coordinates[sample];
            }
            double movement = 0;
            for (size_t level = 0; level < levels; ++level) {
                if (!(masses[level] > 0)) {
                    throw std::runtime_error("TurboQuant sphere codebook has an empty cell");
                }
                const double updated = sums[level] / masses[level];
                movement = std::max(movement, std::abs(updated - centers[level]));
                centers[level] = updated;
            }
            if (movement < 1e-12) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            throw std::runtime_error("TurboQuant sphere codebook did not converge");
        }
        for (size_t level = 0; level < levels; ++level) {
            centroids_[level] = static_cast<float>(centers[level]);
        }
    }

    static float readFloat(const unsigned char *bytes) {
        float value;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    static void writeFloat(unsigned char *bytes, float value) {
        std::memcpy(bytes, &value, sizeof(value));
    }
    static void requireFinite(const float *input, size_t dim) {
        if (!input) {
            throw std::invalid_argument("TurboQuant input is null");
        }
        for (size_t i = 0; i < dim; ++i) {
            if (!std::isfinite(input[i])) {
                throw std::invalid_argument("TurboQuant input must be finite");
            }
        }
    }
    static float checkedFloat(float value) {
        if (!std::isfinite(value)) {
            throw std::overflow_error("TurboQuant FP32 arithmetic overflow");
        }
        return value;
    }
    void writeCode(unsigned char *bytes, size_t coordinate, unsigned code) const {
        const size_t start = coordinate * code_bits_;
        for (unsigned bit = 0; bit < code_bits_; ++bit) {
            bytes[(start + bit) / 8] |= ((code >> bit) & 1U) << ((start + bit) % 8);
        }
    }
    unsigned readCode(const unsigned char *bytes, size_t coordinate) const {
        const size_t start = coordinate * code_bits_;
        unsigned code = 0;
        for (unsigned bit = 0; bit < code_bits_; ++bit) {
            code |= ((bytes[(start + bit) / 8] >> ((start + bit) % 8)) & 1U) << bit;
        }
        return code;
    }
    turboquant_detail::PackedPairDot storedPairKernel() const {
        if (execution_ == TurboQuantExecution::Dispatch) {
#if defined(CPU_FEATURES_ARCH_AARCH64) && defined(__ARM_NEON)
            if (dim_ >= 8 && cpu_features::GetAarch64Info().features.asimd) {
                if (bits_ == 2)
                    return turboquant_detail::packedPairDotNEON<2>;
                if (bits_ == 3)
                    return turboquant_detail::packedPairDotNEON<3>;
            }
#endif
        }
        if (bits_ == 2)
            return turboquant_detail::packedPairDot<2>;
        if (bits_ == 3)
            return turboquant_detail::packedPairDot<3>;
        return turboquant_detail::packedPairDot<4>;
    }
    float residualScale() const {
        return std::sqrt(std::numbers::pi_v<float> / 2.0f) / static_cast<float>(dim_);
    }
    float mseScale(float norm, double unit_code_dot, double code_norm_squared) const {
        double scale = norm;
        if (scale_policy_ == TurboQuantScalePolicy::LeastSquares ||
            scale_policy_ == TurboQuantScalePolicy::ProjectionMatch) {
            if (!(unit_code_dot > 0) || !std::isfinite(unit_code_dot))
                throw std::overflow_error("TurboQuant scale requires a finite positive u.c");
        }
        if (scale_policy_ == TurboQuantScalePolicy::LeastSquares ||
            scale_policy_ == TurboQuantScalePolicy::NormMatch) {
            if (!(code_norm_squared > 0) || !std::isfinite(code_norm_squared))
                throw std::overflow_error("TurboQuant scale requires a finite positive c.c");
        }
        if (scale_policy_ == TurboQuantScalePolicy::LeastSquares)
            scale = double(norm) * unit_code_dot / code_norm_squared;
        else if (scale_policy_ == TurboQuantScalePolicy::NormMatch)
            scale = double(norm) / std::sqrt(code_norm_squared);
        else if (scale_policy_ == TurboQuantScalePolicy::ProjectionMatch)
            scale = double(norm) / unit_code_dot;
        if (!(scale > 0) || !std::isfinite(scale))
            throw std::overflow_error("TurboQuant scale must be finite and positive");
        const float stored_scale = checkedFloat(static_cast<float>(scale));
        if (!(stored_scale > 0))
            throw std::overflow_error("TurboQuant scale underflows FP32");
        return stored_scale;
    }

public:
    TurboQuant(std::shared_ptr<VecSimAllocator> allocator, size_t dim, unsigned bits, uint64_t seed,
               TurboQuantVariant variant = TurboQuantVariant::InnerProduct,
               TurboQuantExecution execution = TurboQuantExecution::Dispatch,
               TurboQuantScalePolicy scale_policy = TurboQuantScalePolicy::Baseline,
               TurboQuantBiasPolicy bias_policy = TurboQuantBiasPolicy::None,
               std::span<const float> mean = {}, bool initialize = true)
        : allocator_(std::move(allocator)),
          dim_(validatedDim(allocator_, dim, bits, variant, execution, scale_policy, bias_policy, mean)),
          bits_(bits), seed_(seed),
          variant_(variant), execution_(execution), scale_policy_(scale_policy),
          bias_policy_(bias_policy),
          code_bits_(variant_ == TurboQuantVariant::MSE ? bits_ : bits_ - 1),
          code_bytes_((dim_ * code_bits_ + 7) / 8),
          sign_bytes_(variant_ == TurboQuantVariant::MSE ? 0 : (dim_ + 7) / 8),
          rotation_(dim_ * dim_, allocator_),
          residual_matrix_(variant_ == TurboQuantVariant::MSE ? 0 : dim_ * dim_, allocator_),
          centroids_(size_t{1} << code_bits_, allocator_), bias_mean_(allocator_),
          bias_rotated_mean_(allocator_) {
        if (initialize) {
            initializeTransforms(seed);
            initializeCentroids();
        }
        initializeBiasMean(mean);
        if (execution_ == TurboQuantExecution::Dispatch && variant_ == TurboQuantVariant::MSE) {
            if (bits_ == 2) {
                packed_dot_ = turboquant_detail::packedDot<2>;
            } else if (bits_ == 3) {
                packed_dot_ = turboquant_detail::packedDot<3>;
            }
#if defined(CPU_FEATURES_ARCH_AARCH64) && defined(__ARM_NEON)
            if (cpu_features::GetAarch64Info().features.asimd) {
                if (dim_ >= 4) {
                    row_dot_ = turboquant_detail::rowDotNEON;
                }
                if (dim_ >= 8) {
                    if (bits_ == 2) {
                        packed_dot_ = turboquant_detail::packedDotNEON<2>;
                    } else if (bits_ == 3) {
                        packed_dot_ = turboquant_detail::packedDotNEON<3>;
                    }
                }
            }
#endif
#ifdef VECSIM_TURBOQUANT_HAS_SVE
            if (dim_ >= 8 && cpu_features::GetAarch64Info().features.sve2) {
                if (bits_ == 2)
                    packed_dot_ = turboquant_detail::packedDotSVE<2>;
                else if (bits_ == 3)
                    packed_dot_ = turboquant_detail::packedDotSVE<3>;
            }
#endif
        }
        if (variant_ == TurboQuantVariant::MSE)
            stored_pair_dot_ = storedPairKernel();
    }

    // Allocations and all validation happen before publication. Readers are excluded while
    // publishMean swaps all three fitted terms; the codec address and dispatch remain stable.
    struct PreparedMean {
        vecsim_stl::vector<float> mean;
        vecsim_stl::vector<float> rotated;
        float normSquared = 0;
        explicit PreparedMean(const std::shared_ptr<VecSimAllocator> &allocator)
            : mean(allocator), rotated(allocator) {}
    };
    PreparedMean prepareMean(std::span<const float> mean) const {
        if (!hasBias() || mean.size() != dim_)
            throw std::invalid_argument("TurboQuant mean dimension mismatch");
        PreparedMean fitted(allocator_);
        fitted.mean.assign(mean.begin(), mean.end());
        fitted.rotated.resize(dim_);
        double norm = 0;
        for (float value : mean) {
            if (!std::isfinite(value))
                throw std::invalid_argument("TurboQuant mean must be finite");
            norm += double(value) * value;
        }
        fitted.normSquared = checkedFloat(static_cast<float>(norm));
        for (size_t row = 0; row < dim_; ++row) {
            double dot = 0;
            for (size_t col = 0; col < dim_; ++col)
                dot += double(rotation_[row * dim_ + col]) * mean[col];
            fitted.rotated[row] = checkedFloat(static_cast<float>(dot));
        }
        return fitted;
    }
    void publishMean(PreparedMean &&fitted) noexcept {
        assert(fitted.mean.size() == dim_ && fitted.rotated.size() == dim_);
        bias_mean_.swap(fitted.mean);
        bias_rotated_mean_.swap(fitted.rotated);
        bias_mean_norm_squared_ = fitted.normSquared;
    }
    uint64_t seed() const { return seed_; }
    float meanNormSquared() const { return bias_mean_norm_squared_; }
#ifdef BUILD_TESTS
    // V6 owns the fitted floats. Seed is metadata, never a substitute for those floats.
    void saveFitted(std::ostream &out) const {
        const uint32_t header[] = {1, bits_, 1, 3, 1}; // schema, MSE, projection-match, mean-error
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        out.write(reinterpret_cast<const char *>(&seed_), sizeof(seed_));
        auto write = [&](const auto &values) {
            uint64_t count = values.size();
            out.write(reinterpret_cast<const char *>(&count), sizeof(count));
            out.write(reinterpret_cast<const char *>(values.data()), count * sizeof(float));
        };
        write(rotation_); write(centroids_); write(bias_mean_); write(bias_rotated_mean_);
        out.write(reinterpret_cast<const char *>(&bias_mean_norm_squared_), sizeof(float));
        if (!out) throw std::runtime_error("Cannot write TurboQuant fitted state");
    }
    static std::shared_ptr<TurboQuant> loadFitted(std::istream &in,
                                                 const std::shared_ptr<VecSimAllocator> &allocator,
                                                 size_t dim, unsigned bits) {
        // Check the dimension and complete payload before allocating based on file contents.
        constexpr size_t limit = std::numeric_limits<size_t>::max() / (4 * sizeof(double));
        if (dim < 2 || dim > limit / dim || (bits != 2 && bits != 3))
            throw std::runtime_error("Invalid TurboQuant fitted dimension");
        uint32_t header[5]{};
        uint64_t seed = 0;
        in.read(reinterpret_cast<char *>(header), sizeof(header));
        in.read(reinterpret_cast<char *>(&seed), sizeof(seed));
        if (!in || header[0] != 1 || header[1] != bits || header[2] != 1 ||
            header[3] != 3 || header[4] != 1)
            throw std::runtime_error("Unsupported TurboQuant fitted policy");
        const auto position = in.tellg();
        in.seekg(0, std::ios::end);
        const auto end = in.tellg();
        in.seekg(position);
        const uint64_t minimum = 4 * sizeof(uint64_t) + sizeof(float) +
            (uint64_t(dim) * dim + (uint64_t(1) << bits) + 2 * dim) * sizeof(float);
        if (!in || position < 0 || end < position || uint64_t(end - position) < minimum)
            throw std::runtime_error("Truncated TurboQuant fitted state");
        vecsim_stl::vector<float> zero(dim, 0.0f, allocator);
        auto codec = std::allocate_shared<TurboQuant>(VecsimSTLAllocator<TurboQuant>(allocator),
            allocator, dim, bits, seed, TurboQuantVariant::MSE, TurboQuantExecution::Dispatch,
            TurboQuantScalePolicy::ProjectionMatch, TurboQuantBiasPolicy::MeanError,
            std::span<const float>(zero.data(), zero.size()), false);
        auto read = [&](auto &values) {
            uint64_t count = 0;
            in.read(reinterpret_cast<char *>(&count), sizeof(count));
            if (!in || count != values.size())
                throw std::runtime_error("TurboQuant fitted length mismatch");
            in.read(reinterpret_cast<char *>(values.data()), count * sizeof(float));
            if (!in) throw std::runtime_error("Truncated TurboQuant fitted state");
            for (float value : values)
                if (!std::isfinite(value))
                    throw std::runtime_error("Nonfinite TurboQuant fitted state");
        };
        read(codec->rotation_); read(codec->centroids_); read(codec->bias_mean_);
        read(codec->bias_rotated_mean_);
        in.read(reinterpret_cast<char *>(&codec->bias_mean_norm_squared_), sizeof(float));
        if (!in || !std::isfinite(codec->bias_mean_norm_squared_))
            throw std::runtime_error("Invalid TurboQuant fitted mean norm");
        for (size_t i = 0; i < codec->centroids_.size(); ++i)
            if (std::abs(codec->centroids_[i]) > 1 ||
                (i && codec->centroids_[i] <= codec->centroids_[i - 1]))
                throw std::runtime_error("Invalid TurboQuant fitted centroids");
        // Validate the relationship between persisted derived terms and the exact stored mean.
        const auto expected = codec->prepareMean(codec->biasMean());
        if (expected.normSquared != codec->bias_mean_norm_squared_ ||
            !std::equal(expected.rotated.begin(), expected.rotated.end(),
                        codec->bias_rotated_mean_.begin()))
            throw std::runtime_error("Inconsistent TurboQuant fitted mean terms");
        for (size_t row = 0; row < dim; ++row) {
            double squared = 0;
            for (size_t col = 0; col < dim; ++col) {
                const double value = codec->rotation_[row * dim + col];
                squared += value * value;
            }
            if (std::abs(squared - 1.0) > 1e-4)
                throw std::runtime_error("Invalid TurboQuant fitted rotation");
        }
        return codec;
    }
#endif

    size_t dim() const { return dim_; }
    unsigned bits() const { return bits_; }
    TurboQuantVariant variant() const { return variant_; }
    TurboQuantExecution execution() const { return execution_; }
    TurboQuantScalePolicy scalePolicy() const { return scale_policy_; }
    TurboQuantBiasPolicy biasPolicy() const { return bias_policy_; }
    bool hasBias() const { return bias_policy_ == TurboQuantBiasPolicy::MeanError; }
    const char *biasPolicyName() const { return hasBias() ? "mean-error" : "none"; }
    const char *biasPolicyFormula() const { return hasBias() ? "s*(c.Rmean)-r.mean" : "0"; }
    unsigned biasReductionTermsPerCoordinate() const { return hasBias() ? 2 : 0; }
    std::span<const float> biasMean() const { return {bias_mean_.data(), bias_mean_.size()}; }
    std::span<const float> biasRotatedMean() const {
        return {bias_rotated_mean_.data(), bias_rotated_mean_.size()};
    }
    // Adapters must center with the same original-coordinate mean used for the offset.
    void validateBiasMean(std::span<const float> mean) const {
        if (hasBias() &&
            (mean.size() != dim_ || !std::equal(mean.begin(), mean.end(), bias_mean_.begin())))
            throw std::invalid_argument("TurboQuant centering mean does not match bias mean");
    }
    float biasOffset(const void *stored) const {
        if (!stored)
            throw std::invalid_argument("TurboQuant storage is null");
        return hasBias() ? readFloat(static_cast<const unsigned char *>(stored) + code_bytes_ +
                                     sizeof(float)) : 0.0f;
    }
    const char *scalePolicyName() const {
        switch (scale_policy_) {
        case TurboQuantScalePolicy::Baseline: return "baseline";
        case TurboQuantScalePolicy::LeastSquares: return "least-squares";
        case TurboQuantScalePolicy::NormMatch: return "norm-match";
        case TurboQuantScalePolicy::ProjectionMatch: return "projection-match";
        }
        throw std::logic_error("Invalid TurboQuant scale policy");
    }
    // n=||r||, u=R*r/n, c=the unchanged centroid decode, r=x-mean.
    const char *scalePolicyFormula() const {
        switch (scale_policy_) {
        case TurboQuantScalePolicy::Baseline: return "n";
        case TurboQuantScalePolicy::LeastSquares: return "n*(u.c)/(c.c)";
        case TurboQuantScalePolicy::NormMatch: return "n/sqrt(c.c)";
        case TurboQuantScalePolicy::ProjectionMatch: return "n/(u.c)";
        }
        throw std::logic_error("Invalid TurboQuant scale policy");
    }
    unsigned scaleReductionTermsPerCoordinate() const {
        return scale_policy_ == TurboQuantScalePolicy::Baseline ? 0
             : scale_policy_ == TurboQuantScalePolicy::LeastSquares ? 2 : 1;
    }
    const char *queryKernelName() const { return row_dot_ ? "neon-raw-dot" : "scalar-reference"; }
    const char *scoringKernelName() const {
#ifdef VECSIM_TURBOQUANT_HAS_SVE
        if (packed_dot_ == turboquant_detail::packedDotSVE<2> ||
            packed_dot_ == turboquant_detail::packedDotSVE<3>)
            return "sve2-128-packed-centroids-with-neon-fallback";
#endif
#if defined(CPU_FEATURES_ARCH_AARCH64) && defined(__ARM_NEON)
        if (packed_dot_ == turboquant_detail::packedDotNEON<2> ||
            packed_dot_ == turboquant_detail::packedDotNEON<3>) {
            return "neon-packed-centroids";
        }
#endif
        return packed_dot_ ? "scalar-packed-centroids" : "scalar-reference";
    }
    // Separate byte-rounded code and sign planes, followed by two unaligned floats:
    // [ceil(d*(b-1)/8) codes][ceil(d/8) signs][norm][residual norm].
    // MSE uses all b bits for centroids: [ceil(d*b/8) codes][scale].
    // Baseline scale is the exact input norm; other research policies reuse this float.
    // MeanError appends one float delta after the MSE scale.
    size_t storageBytes() const {
        return code_bytes_ + sign_bytes_ +
               ((variant_ == TurboQuantVariant::MSE ? 1 : 2) + (hasBias() ? 1 : 0)) * sizeof(float);
    }
    // MSE: [Pi*q: d floats]. InnerProduct additionally stores [A*Pi*q: d floats].
    size_t queryBytes() const {
        return (variant_ == TurboQuantVariant::MSE ? 1 : 2) * dim_ * sizeof(float);
    }
    size_t sharedDataBytes() const {
        return (rotation_.size() + residual_matrix_.size() + centroids_.size() +
                bias_mean_.size() + bias_rotated_mean_.size()) * sizeof(float);
    }
    std::span<const float> rotation() const { return {rotation_.data(), rotation_.size()}; }
    std::span<const float> residualMatrix() const {
        return {residual_matrix_.data(), residual_matrix_.size()};
    }
    std::span<const float> centroids() const { return {centroids_.data(), centroids_.size()}; }

    void encode(const float *input, void *stored) const {
        requireFinite(input, dim_);
        if (!stored) {
            throw std::invalid_argument("TurboQuant storage is null");
        }
        float norm = 0;
        double residual_mean_dot = 0;
        for (size_t i = 0; i < dim_; ++i) {
            norm = std::hypot(norm, input[i]);
            if (hasBias())
                residual_mean_dot += double(input[i]) * bias_mean_[i];
        }
        checkedFloat(norm);
        auto *bytes = static_cast<unsigned char *>(stored);
        if (norm == 0) {
            std::memset(bytes, 0, storageBytes());
            return;
        }
        // Consume input before writing output, allowing callers to reuse the input buffer.
        vecsim_stl::vector<float> unit(dim_, allocator_);
        for (size_t i = 0; i < dim_; ++i) {
            unit[i] = input[i] / norm;
        }
        std::memset(bytes, 0, storageBytes());
        vecsim_stl::vector<float> residual(variant_ == TurboQuantVariant::MSE ? 0 : dim_,
                                           allocator_);
        float residual_norm_squared = 0;
        double unit_code_dot = 0, code_norm_squared = 0, code_mean_dot = 0;
        for (size_t row = 0; row < dim_; ++row) {
            float rotated = 0;
            for (size_t col = 0; col < dim_; ++col) {
                rotated += rotation_[row * dim_ + col] * unit[col];
            }
            unsigned code = 0;
            while (code + 1 < centroids_.size() &&
                   rotated > (centroids_[code] + centroids_[code + 1]) * 0.5f) {
                ++code;
            }
            writeCode(bytes, row, code);
            if (hasBias())
                code_mean_dot += double(centroids_[code]) * bias_rotated_mean_[row];
            if (scale_policy_ != TurboQuantScalePolicy::Baseline) {
                const double centroid = centroids_[code];
                if (scale_policy_ != TurboQuantScalePolicy::NormMatch)
                    unit_code_dot += double(rotated) * centroid;
                if (scale_policy_ != TurboQuantScalePolicy::ProjectionMatch)
                    code_norm_squared += centroid * centroid;
            }
            if (variant_ == TurboQuantVariant::InnerProduct) {
                residual[row] = rotated - centroids_[code];
                residual_norm_squared += residual[row] * residual[row];
            }
        }
        if (variant_ == TurboQuantVariant::MSE) {
            // Baseline retains its original FP32 arithmetic and scalar exactly.
            const float scale = scale_policy_ == TurboQuantScalePolicy::Baseline
                                    ? norm : mseScale(norm, unit_code_dot, code_norm_squared);
            writeFloat(bytes + code_bytes_, scale);
            if (hasBias()) {
                const float delta = checkedFloat(
                    static_cast<float>(double(scale) * code_mean_dot - residual_mean_dot));
                writeFloat(bytes + code_bytes_ + sizeof(float), delta);
            }
            return;
        }
        for (size_t row = 0; row < dim_; ++row) {
            float projection = 0;
            for (size_t col = 0; col < dim_; ++col) {
                projection += residual_matrix_[row * dim_ + col] * residual[col];
            }
            if (projection >= 0) {
                bytes[code_bytes_ + row / 8] |= 1U << (row % 8);
            }
        }
        writeFloat(bytes + code_bytes_ + sign_bytes_, norm);
        writeFloat(bytes + code_bytes_ + sign_bytes_ + sizeof(float),
                   std::sqrt(residual_norm_squared));
    }

    void prepareQuery(const float *input, void *query) const {
        requireFinite(input, dim_);
        if (!query) {
            throw std::invalid_argument("TurboQuant query buffer is null");
        }
        auto *bytes = static_cast<unsigned char *>(query);
        // Consume input before writing output, allowing input and query to overlap.
        vecsim_stl::vector<float> rotated(dim_, allocator_);
        for (size_t row = 0; row < dim_; ++row) {
            float value = 0;
            if (row_dot_) {
                value = row_dot_(rotation_.data() + row * dim_, input, dim_);
            } else {
                for (size_t col = 0; col < dim_; ++col) {
                    value += rotation_[row * dim_ + col] * input[col];
                }
            }
            rotated[row] = checkedFloat(value);
        }
        for (size_t row = 0; row < dim_; ++row) {
            writeFloat(bytes + row * sizeof(float), rotated[row]);
        }
        if (variant_ == TurboQuantVariant::MSE) {
            return;
        }
        for (size_t row = 0; row < dim_; ++row) {
            float value = 0;
            for (size_t col = 0; col < dim_; ++col) {
                value += residual_matrix_[row * dim_ + col] * rotated[col];
            }
            writeFloat(bytes + (dim_ + row) * sizeof(float), checkedFloat(value));
        }
    }

    float distance(const void *stored, const void *query) const {
        if (!stored || !query) {
            throw std::invalid_argument("TurboQuant distance buffer is null");
        }
        const auto *bytes = static_cast<const unsigned char *>(stored);
        const auto *query_bytes = static_cast<const unsigned char *>(query);
        const float norm = readFloat(bytes + code_bytes_ + sign_bytes_);
        if (norm == 0) {
            return 1.0f;
        }
        if (packed_dot_) {
            const float score = checkedFloat(
                1.0f - norm * packed_dot_(bytes, query_bytes, centroids_.data(), dim_));
            return hasBias() ? checkedFloat(score + biasOffset(stored)) : score;
        }
        float mse_dot = 0;
        float residual_dot = 0;
        for (size_t i = 0; i < dim_; ++i) {
            mse_dot += readFloat(query_bytes + i * sizeof(float)) * centroids_[readCode(bytes, i)];
            if (variant_ == TurboQuantVariant::InnerProduct) {
                const float sign = ((bytes[code_bytes_ + i / 8] >> (i % 8)) & 1U) ? 1.0f : -1.0f;
                residual_dot += readFloat(query_bytes + (dim_ + i) * sizeof(float)) * sign;
            }
        }
        if (variant_ == TurboQuantVariant::MSE) {
            const float score = checkedFloat(1.0f - norm * mse_dot);
            return hasBias() ? checkedFloat(score + biasOffset(stored)) : score;
        }
        const float gamma = readFloat(bytes + code_bytes_ + sign_bytes_ + sizeof(float));
        return checkedFloat(1.0f - norm * (mse_dot + gamma * residualScale() * residual_dot));
    }

    // Experimental symmetric MSE graph estimator. Codes represent x - mean, so both
    // cross terms and the original-coordinate mean norm belong in the inner product.
    // None accepts caller-precomputed mean terms. MeanError uses its bound terms and
    // requires the optional arguments to be absent. No rows or scratch allocations are retained.
    float storedDistance(const void *a, const void *b, std::span<const float> rotated_mean = {},
                         float mean_norm_squared = 0) const {
        if (variant_ != TurboQuantVariant::MSE)
            throw std::invalid_argument("TurboQuant stored pairs require the MSE variant");
        if (!a || !b || (!rotated_mean.empty() && rotated_mean.size() != dim_) ||
            !std::isfinite(mean_norm_squared) || mean_norm_squared < 0)
            throw std::invalid_argument("Invalid TurboQuant stored-pair configuration");
        if (hasBias()) {
            if (!rotated_mean.empty() || mean_norm_squared != 0)
                throw std::invalid_argument("TurboQuant bias stored pairs use bound mean terms");
            rotated_mean = biasRotatedMean();
            mean_norm_squared = bias_mean_norm_squared_;
        }
        const auto *bytes_a = static_cast<const unsigned char *>(a);
        const auto *bytes_b = static_cast<const unsigned char *>(b);
        const float norm_a = readFloat(bytes_a + code_bytes_);
        const float norm_b = readFloat(bytes_b + code_bytes_);
        const float dot =
            stored_pair_dot_(bytes_a, bytes_b, centroids_.data(), dim_, norm_a, norm_b,
                             rotated_mean.empty() ? nullptr : rotated_mean.data());
        const float score = checkedFloat(1.0f - (dot + mean_norm_squared));
        return hasBias() ? checkedFloat(score + (biasOffset(a) + biasOffset(b))) : score;
    }
    const char *storedScoringKernelName() const {
        if (variant_ != TurboQuantVariant::MSE)
            return "unsupported-inner-product-variant";
#if defined(CPU_FEATURES_ARCH_AARCH64) && defined(__ARM_NEON)
        const auto kernel = stored_pair_dot_;
        if (kernel == turboquant_detail::packedPairDotNEON<2> ||
            kernel == turboquant_detail::packedPairDotNEON<3>)
            return "neon-packed-pair-centroids";
#endif
        return bits_ == 4 ? "scalar-pair-centroids" : "scalar-packed-pair-centroids";
    }

    // Decoded estimator in rotated coordinates; MSE centroids are never renormalized.
    void decodeRotated(const void *stored, float *out) const {
        if (!stored || !out) {
            throw std::invalid_argument("TurboQuant decode buffer is null");
        }
        const auto *bytes = static_cast<const unsigned char *>(stored);
        const float norm = readFloat(bytes + code_bytes_ + sign_bytes_);
        if (variant_ == TurboQuantVariant::MSE) {
            for (size_t col = 0; col < dim_; ++col) {
                out[col] = norm == 0 ? 0.0f : checkedFloat(norm * centroids_[readCode(bytes, col)]);
            }
            return;
        }
        const float gamma = readFloat(bytes + code_bytes_ + sign_bytes_ + sizeof(float));
        for (size_t col = 0; col < dim_; ++col) {
            float correction = 0;
            for (size_t row = 0; row < dim_; ++row) {
                const float sign =
                    ((bytes[code_bytes_ + row / 8] >> (row % 8)) & 1U) ? 1.0f : -1.0f;
                correction += residual_matrix_[row * dim_ + col] * sign;
            }
            out[col] = checkedFloat(
                norm * (centroids_[readCode(bytes, col)] + gamma * residualScale() * correction));
        }
    }
};

} // namespace vecsim::quantization
