/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/utils/alignment.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sq8_l2_detail {

using Wide = unsigned __int128;
static_assert(sizeof(size_t) <= sizeof(uint64_t));
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24);
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53);

struct FloatParts {
    uint32_t bits;
    uint32_t significand;
    int exponent;
    bool negative;

    explicit FloatParts(float value) : bits(std::bit_cast<uint32_t>(value)) {
        const unsigned field = (bits >> 23) & 255;
        significand = (bits & 0x7fffff) | (field ? 0x800000 : 0);
        exponent = field ? int(field) - 150 : -149;
        negative = (bits >> 31) != 0;
    }
    bool finite() const { return (bits & 0x7f800000) != 0x7f800000; }
    bool nan() const { return !finite() && (bits & 0x7fffff); }
    bool subnormal() const { return (bits & 0x7fffffff) && !(bits & 0x7f800000); }
};

// All FP32 values are integer multiples of 2^-149, so their products use a 2^-298 unit.
// Each positive/negative partial sum is below n*(|min| + 255*|delta| + |y|)^2 < 2^337
// even for a 64-bit dimension. Ten limbs cover exponents [-298, 341] without allocation.
struct Accumulator {
    static constexpr int fractional_bits = 298;
    std::array<uint64_t, 10> words{};

    void addWord(size_t index, uint64_t value) {
        while (value) {
            assert(index < words.size());
            const uint64_t old = words[index];
            words[index++] += value;
            value = words[index - 1] < old;
        }
    }

    void add(Wide coefficient, int exponent) {
        const unsigned shift = unsigned(exponent + fractional_bits);
        assert(exponent >= -fractional_bits);
        const size_t index = shift / 64;
        const unsigned offset = shift % 64;
        const uint64_t low = uint64_t(coefficient), high = uint64_t(coefficient >> 64);
        addWord(index, low << offset);
        addWord(index + 1, high << offset);
        if (offset) {
            addWord(index + 1, low >> (64 - offset));
            addWord(index + 2, high >> (64 - offset));
        }
    }

    void subtract(const Accumulator &other) {
        uint64_t borrow = 0;
        for (size_t i = 0; i < words.size(); ++i) {
            const Wide subtrahend = Wide(other.words[i]) + borrow;
            const uint64_t old = words[i];
            words[i] = old - uint64_t(subtrahend);
            borrow = Wide(old) < subtrahend;
        }
        assert(!borrow); // The exact sum of squared residuals cannot be negative.
    }

    bool bit(unsigned index) const { return (words[index / 64] >> (index % 64)) & 1; }

    bool anyBelow(unsigned count) const {
        for (size_t i = 0; i < count / 64; ++i) {
            if (words[i])
                return true;
        }
        const unsigned remainder = count % 64;
        return remainder && (words[count / 64] & ((uint64_t{1} << remainder) - 1));
    }

    float rounded() const {
        int highest = -1;
        for (size_t i = words.size(); i-- > 0;) {
            if (words[i]) {
                highest = int(i * 64 + 63 - std::countl_zero(words[i]));
                break;
            }
        }
        if (highest < 0)
            return 0.0f;
        // Subnormals keep a fixed 2^-149 quantum; normals keep 24 significant bits.
        const unsigned shift = unsigned(std::max(149, highest - 23));
        const size_t index = shift / 64;
        const unsigned offset = shift % 64;
        uint64_t top = words[index] >> offset;
        if (offset && index + 1 < words.size())
            top |= words[index + 1] << (64 - offset);
        uint32_t significand = uint32_t(top);
        if (bit(shift - 1) && (anyBelow(shift - 1) || (significand & 1)))
            ++significand;
        if (highest < 172)
            return std::bit_cast<float>(
                significand); // Includes rounding up to the smallest normal.
        if (significand == 0x1000000) {
            significand >>= 1;
            ++highest;
        }
        const int exponent = highest - fractional_bits;
        if (exponent > 127)
            return std::bit_cast<float>(uint32_t{0x7f800000});
        return std::bit_cast<float>((uint32_t(exponent + 127) << 23) | (significand & 0x7fffff));
    }
};

inline float exact(const uint8_t *codes, const float *query, size_t dim, float min_val,
                   float delta) {
    const FloatParts m(min_val), d(delta);
    if (!m.finite() || !d.finite())
        return std::bit_cast<float>(uint32_t{0x7fc00000});
    Accumulator positive, negative;
    Wide code_sum = 0, code_squares = 0;
    bool infinite_query = false;
    for (size_t i = 0; i < dim; ++i) {
        const FloatParts y(load_unaligned<float>(query + i));
        if (y.nan())
            return std::bit_cast<float>(uint32_t{0x7fc00000});
        if (!y.finite()) {
            infinite_query = true;
            continue;
        }
        const uint32_t q = codes[i];
        code_sum += q;
        code_squares += q * q;
        positive.add(Wide(y.significand) * y.significand, 2 * y.exponent);
        (m.negative == y.negative ? negative : positive)
            .add(Wide{2} * m.significand * y.significand, m.exponent + y.exponent);
        (d.negative == y.negative ? negative : positive)
            .add(Wide{2} * q * d.significand * y.significand, d.exponent + y.exponent);
    }
    if (infinite_query)
        return std::bit_cast<float>(uint32_t{0x7f800000});
    // Expanding the identity is safe here: every product and addition is exact integer arithmetic.
    positive.add(Wide(m.significand) * m.significand * dim, 2 * m.exponent);
    positive.add(Wide(d.significand) * d.significand * code_squares, 2 * d.exponent);
    (m.negative == d.negative ? positive : negative)
        .add(Wide{2} * m.significand * d.significand * code_sum, m.exponent + d.exponent);
    positive.subtract(negative);
    return positive.rounded();
}

inline bool certified(const uint8_t *codes, const float *query, size_t dim, float min_val,
                      float delta, float &result) {
#if defined(__FAST_MATH__)
    return false;
#else
    if (dim > (size_t{1} << 24) || std::fegetround() != FE_TONEAREST)
        return false;
    const FloatParts m(min_val), d(delta);
    if (!m.finite() || !d.finite() || m.subnormal() || d.subnormal())
        return false;
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const float value = load_unaligned<float>(query + i);
        const FloatParts y(value);
        if (!y.finite() || y.subnormal())
            return false;
        // With at most 28 exponent bits between two FP32 values, their sum/difference fits
        // within FP64's 53 significant bits. Zeros are exact regardless of the other exponent.
        if (m.significand && y.significand && std::abs(m.exponent - y.exponent) > 28)
            return false;
        const double shifted_query = double(min_val) - double(value);
        const double product = double(delta) * codes[i]; // At most 24 + 8 significant bits.
        const double residual = shifted_query + product;
        sum += residual * residual;
    }
    if (sum == 0.0) {
        result = 0.0f;
        return true;
    }
    // Each residual has one rounding, its square has three in total, and summation adds n-1.
    // For n <= 2^24 the standard gamma_(n+2) bound, expressed relative to this computed sum,
    // is strictly below 4*(n+4)*epsilon. The slack also covers rounding this bound itself.
    // FP32-derived residuals and their squares cannot underflow or overflow FP64 here.
    const double error = (4.0 * (double(dim) + 4.0) * std::numeric_limits<double>::epsilon()) * sum;
    const double lower = std::nextafter(sum - error, -std::numeric_limits<double>::infinity());
    const double upper = std::nextafter(sum + error, std::numeric_limits<double>::infinity());
    // Integer rounding handles FP32 under/overflow, also when the host flushes subnormals.
    if (lower < double(std::numeric_limits<float>::min()) ||
        upper > double(std::numeric_limits<float>::max()))
        return false;
    const float low = float(lower), high = float(upper);
    if (std::bit_cast<uint32_t>(low) != std::bit_cast<uint32_t>(high))
        return false;
    result = low;
    return true;
#endif
}

} // namespace sq8_l2_detail
