/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#include "TQ.h"

#include "VecSim/spaces/IP/IP_SVE_FP32.h"

namespace spaces {

#include "implementation_chooser.h"

template <bool partial_chunk, unsigned char additional_steps>
float FP32_InnerProductSIMD_TQ_SVE(const void *lhs, const void *rhs, size_t dim) {
    return 1.0f - FP32_InnerProductSIMD_SVE<partial_chunk, additional_steps>(lhs, rhs, dim);
}

template <bool partial_chunk, unsigned char additional_steps>
float FP32_SumSquaresSIMD_TQ_SVE(const void *values, size_t dim) {
    return FP32_InnerProductSIMD_TQ_SVE<partial_chunk, additional_steps>(values, values, dim);
}

tq_inner_product_func_t Choose_FP32_InnerProduct_implementation_TQ_SVE(size_t dim) {
    tq_inner_product_func_t ret_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_func, FP32_InnerProductSIMD_TQ_SVE, dim, svcntw);
    return ret_func;
}

tq_sum_squares_func_t Choose_FP32_SumSquares_implementation_TQ_SVE(size_t dim) {
    tq_sum_squares_func_t ret_func;
    CHOOSE_SVE_IMPLEMENTATION(ret_func, FP32_SumSquaresSIMD_TQ_SVE, dim, svcntw);
    return ret_func;
}

#include "implementation_chooser_cleanup.h"

} // namespace spaces
