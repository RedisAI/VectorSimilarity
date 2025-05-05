/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include "VecSim/vec_sim_common.h"
#include "VecSim/types/bfloat16.h"
#include "VecSim/types/float16.h"
template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {

    static VecSimType get_index_type() { return type; }

    typedef DataType data_t;
    typedef DistType dist_t;
};

using fp32_index_t = IndexType<VecSimType_FLOAT32, float, float>;
using fp64_index_t = IndexType<VecSimType_FLOAT64, double, double>;
using bf16_index_t = IndexType<VecSimType_BFLOAT16, vecsim_types::bfloat16, float>;
using fp16_index_t = IndexType<VecSimType_FLOAT16, vecsim_types::float16, float>;
using int8_index_t = IndexType<VecSimType_INT8, int8_t, float>;
using uint8_index_t = IndexType<VecSimType_UINT8, uint8_t, float>;

#define INDICES   BM_VecSimIndex<index_type_t>::indices
#define QUERIES   BM_VecSimIndex<index_type_t>::queries
#define N_QUERIES BM_VecSimGeneral::n_queries
#define N_VECTORS BM_VecSimGeneral::n_vectors
#define DIM       BM_VecSimGeneral::dim
#define IS_MULTI  BM_VecSimGeneral::is_multi
#define REF_COUNT BM_VecSimIndex<index_type_t>::ref_count
