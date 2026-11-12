/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "VecSim/types/float16.h"
#include "VecSim/types/sq8_a8.h"

// Scalar encoder of the A8 query blob (vecsim_types::sq8_a8_query). It is ISA-neutral, with
// internal linkage, so a vectorized encoder can reuse the per-element step for its tail and must
// produce the same codes bit for bit. The codes depend only on double operations that do not
// change under FP contraction (128 * s is exact), so they match across translation units; the
// constants are plain sums and may differ in the last bits between encoders.
namespace spaces {
namespace quantized {

// The query's step s and offset c from its range. A constant query gets s = 1, which still
// reconstructs it exactly (every code is -128 and every rounding error is 0).
struct SQ8_A8_QueryGrid {
    double step;
    double offset;
};

static inline SQ8_A8_QueryGrid SQ8_A8_QueryGridFromRange(float min_val, float max_val) {
    // In double, so the range of a query spanning most of the float range stays finite.
    const double step = max_val > min_val ? (static_cast<double>(max_val) - min_val) / 255.0 : 1.0;
    return {step, static_cast<double>(min_val) + 128.0 * step};
}

// Running sums of the encoding: P = sum p_i, E = sum e_i and PE = sum p_i * e_i, where
// e_i = v_i - s * p_i is the rounding error in the frame centered on c (v_i = y_i - c is exact).
struct SQ8_A8_QuerySums {
    int64_t P = 0;
    double E = 0;
    double PE = 0;
};

// Encodes one element: p = clamp(floor((y - c) / s + 0.5), -128, 127). The clamp is written as
// max then min with the bound second, which maps NaN to -128 like the SIMD max / min do.
static inline int8_t SQ8_A8_EncodeElement(float y, const SQ8_A8_QueryGrid &grid,
                                          SQ8_A8_QuerySums &sums) {
    const double v = static_cast<double>(y) - grid.offset;
    double code = std::floor(v / grid.step + 0.5);
    code = code > -128.0 ? code : -128.0;
    code = code < 127.0 ? code : 127.0;
    const double err = v - grid.step * code;
    sums.P += static_cast<int64_t>(code);
    sums.E += err;
    sums.PE += code * err;
    return static_cast<int8_t>(code);
}

// Zeroes the padding after the codes and writes the constants. K = s * sum (p_i - P / dim) * e_i
// is formed in one pass as s * (PE - (P / dim) * E). Y_MEAN_IP is written as 0; a mean-centered
// index overwrites it.
static inline void SQ8_A8_FinishQuery(const SQ8_A8_QueryGrid &grid, const SQ8_A8_QuerySums &sums,
                                      size_t dim, void *blob) {
    using a8 = vecsim_types::sq8_a8_query;
    auto *out = static_cast<uint8_t *>(blob);
    const size_t consts_offset = a8::consts_offset(dim);
    std::memset(out + dim, 0, consts_offset - dim);

    const double dim_d = static_cast<double>(dim);
    const double p_sum = static_cast<double>(sums.P);
    const double consts[a8::P] = {
        grid.step,                                        // QD
        grid.step * p_sum,                                // QM
        grid.offset + sums.E / dim_d,                     // QS
        grid.step * (sums.PE - (p_sum / dim_d) * sums.E), // QK
        0.0,                                              // Y_MEAN_IP
    };
    auto *consts_out = out + consts_offset;
    std::memcpy(consts_out, consts, sizeof(consts));
    std::memcpy(consts_out + a8::P * sizeof(double), &sums.P, sizeof(int64_t));
}

// Writes <mean, y> of the uncentered query to the Y_MEAN_IP slot. Each product of two floats is
// exact in double, so only the summation order can differ between encoders.
static inline void SQ8_A8_SetQueryMeanIP(double y_mean_ip, size_t dim, void *blob) {
    using a8 = vecsim_types::sq8_a8_query;
    std::memcpy(static_cast<uint8_t *>(blob) + a8::consts_offset(dim) +
                    a8::Y_MEAN_IP * sizeof(double),
                &y_mean_ip, sizeof(y_mean_ip));
}

// Widens one input element to FP32.
template <typename InputType>
static inline float SQ8_A8_InputValue(const InputType *query, size_t i) {
    if constexpr (std::is_same_v<InputType, vecsim_types::float16>) {
        return vecsim_types::FP16_to_FP32(query[i]);
    } else {
        return query[i];
    }
}

// Writes the full A8 query blob (vecsim_types::sq8_a8_query::blob_bytes_count(dim) bytes) for a
// float or float16 query. FP16 values are widened to FP32 first, so an FP16 query encodes exactly
// as its FP32 widening.
//
// mean is nullptr, or the mean of a mean-centered index (dim FP32 values). Then the query is
// centered in FP32 as y - mean, exactly as the stored vectors are, and <mean, y> of the uncentered
// query is written to Y_MEAN_IP.
template <typename InputType>
static inline void SQ8_A8_EncodeQuery(const InputType *query, const float *mean, size_t dim,
                                      void *blob) {
    static_assert(std::is_same_v<InputType, float> ||
                      std::is_same_v<InputType, vecsim_types::float16>,
                  "The A8 query encoder takes float or float16 input");
    assert(dim > 0);
    const auto value = [query, mean](size_t i) -> float {
        const float y = SQ8_A8_InputValue(query, i);
        return mean ? y - mean[i] : y;
    };

    float min_val = value(0);
    float max_val = min_val;
    for (size_t i = 1; i < dim; i++) {
        const float y = value(i);
        min_val = y < min_val ? y : min_val;
        max_val = y > max_val ? y : max_val;
    }
    const SQ8_A8_QueryGrid grid = SQ8_A8_QueryGridFromRange(min_val, max_val);

    auto *codes = static_cast<int8_t *>(blob);
    SQ8_A8_QuerySums sums;
    for (size_t i = 0; i < dim; i++) {
        codes[i] = SQ8_A8_EncodeElement(value(i), grid, sums);
    }
    SQ8_A8_FinishQuery(grid, sums, dim, blob);

    if (mean) {
        double y_mean_ip = 0;
        for (size_t i = 0; i < dim; i++) {
            y_mean_ip += static_cast<double>(mean[i]) * SQ8_A8_InputValue(query, i);
        }
        SQ8_A8_SetQueryMeanIP(y_mean_ip, dim, blob);
    }
}

// The query of an index without a mean.
template <typename InputType>
static inline void SQ8_A8_EncodeQuery(const InputType *query, size_t dim, void *blob) {
    SQ8_A8_EncodeQuery(query, nullptr, dim, blob);
}

} // namespace quantized
} // namespace spaces
