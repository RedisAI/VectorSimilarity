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

enum IndexTypeIndex {
    INDEX_BF = 0,
    INDEX_BF_UPDATED,
    INDEX_HNSW,
    INDEX_HNSW_UPDATED,
    INDEX_TIERED_HNSW,
    INDEX_SVS,
    INDEX_TIERED_SVS,
    INDEX_SVS_QUANTIZED,
    NUMBER_OF_INDEX_TYPES
};

enum IndexTypeFlags {
    INDEX_TYPE_BF = 1 << 0,
    INDEX_TYPE_BF_UPDATED = 1 << 1,
    INDEX_TYPE_HNSW = 1 << 2,
    INDEX_TYPE_HNSW_UPDATED = 1 << 3,
    INDEX_TYPE_TIERED_HNSW = 1 << 4,
    INDEX_TYPE_SVS = 1 << 5,
    INDEX_TYPE_TIERED_SVS = 1 << 6,
    INDEX_TYPE_SVS_QUANTIZED = 1 << 7
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
