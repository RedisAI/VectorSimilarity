#pragma once
#include "VecSim/vec_sim_common.h"

template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {
    static VecSimType get_index_type() { return type; }
    typedef DataType data_t;
    typedef DistType dist_t;
};

using fp32_index_t = IndexType<VecSimType_FLOAT32, float, float>;
using fp64_index_t = IndexType<VecSimType_FLOAT64, double, double>;

using Offset_t = unsigned short;

#define INDICES   BM_VecSimGeneral::indices
#define QUERIES   BM_VecSimIndex<index_type_t>::queries
#define N_QUERIES BM_VecSimGeneral::n_queries
#define N_VECTORS BM_VecSimGeneral::n_vectors
#define DIM       BM_VecSimGeneral::dim
