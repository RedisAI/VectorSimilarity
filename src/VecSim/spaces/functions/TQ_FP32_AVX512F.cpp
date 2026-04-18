/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "TQ.h"

#include "VecSim/spaces/IP/IP_AVX512F_FP32.h"

namespace spaces {

#include "implementation_chooser.h"

template <unsigned char residual>
float FP32_InnerProductSIMD16_TQ_AVX512F(const void *lhs, const void *rhs, size_t dim) {
    return 1.0f - FP32_InnerProductSIMD16_AVX512<residual>(lhs, rhs, dim);
}

template <unsigned char residual>
float FP32_SumSquaresSIMD16_TQ_AVX512F(const void *values, size_t dim) {
    return FP32_InnerProductSIMD16_TQ_AVX512F<residual>(values, values, dim);
}

tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_AVX512F(size_t dim) {
    tq_inner_product_func_t ret_func;
    CHOOSE_IMPLEMENTATION(ret_func, dim, 16, FP32_InnerProductSIMD16_TQ_AVX512F);
    return ret_func;
}

tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_AVX512F(size_t dim) {
    tq_sum_squares_func_t ret_func;
    CHOOSE_IMPLEMENTATION(ret_func, dim, 16, FP32_SumSquaresSIMD16_TQ_AVX512F);
    return ret_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
