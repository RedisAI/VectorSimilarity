/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "VecSim/spaces/sq8_fp64.h"
#include "VecSim/types/sq8_fp64.h"
#include "VecSim/utils/alignment.h"

namespace spaces {
namespace {
double metadata(const void *blob, size_t dim, size_t index) {
    return load_unaligned<double>(static_cast<const uint8_t *>(blob) + dim +
                                  index * sizeof(double));
}

template <bool L2, bool Asymmetric>
double distance(const void *left, const void *right, size_t dim) {
    const auto *const a = static_cast<const uint8_t *>(left);
    const auto *const b = static_cast<const uint8_t *>(right);
    const double min_a = metadata(left, dim, vecsim_types::sq8_fp64::MIN_VAL);
    const double delta_a = metadata(left, dim, vecsim_types::sq8_fp64::DELTA);
    const double min_b = Asymmetric ? 0.0 : metadata(right, dim, vecsim_types::sq8_fp64::MIN_VAL);
    const double delta_b = Asymmetric ? 0.0 : metadata(right, dim, vecsim_types::sq8_fp64::DELTA);
    double result = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        if constexpr (L2) {
            const double residual =
                Asymmetric
                    ? delta_a * a[i] + (min_a - load_unaligned<double>(b + i * sizeof(double)))
                    : (min_a - min_b) + (delta_a - delta_b) * a[i] +
                          delta_b * (int(a[i]) - int(b[i]));
            result += residual * residual;
        } else {
            const double y = Asymmetric ? load_unaligned<double>(b + i * sizeof(double))
                                        : min_b + delta_b * b[i];
            result += (min_a + delta_a * a[i]) * y;
        }
    }
    return L2 ? result : 1.0 - result;
}
} // namespace

double L2_SQ8FP64(const void *left, const void *right, size_t dim) {
    return distance<true, false>(left, right, dim);
}
double L2_SQ8FP64_FP64(const void *left, const void *right, size_t dim) {
    return distance<true, true>(left, right, dim);
}
double IP_SQ8FP64(const void *left, const void *right, size_t dim) {
    return distance<false, false>(left, right, dim);
}
double IP_SQ8FP64_FP64(const void *left, const void *right, size_t dim) {
    return distance<false, true>(left, right, dim);
}
} // namespace spaces
