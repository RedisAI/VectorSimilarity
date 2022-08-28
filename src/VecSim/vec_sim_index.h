#pragma once
#include "vec_sim_common.h"
#include "vec_sim_interface.h"
#include "query_results.h"
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/spaces/spaces.h"
#include "info_iterator_struct.h"
#include <cassert>
#include <cmath>  //sqrt
#include <limits> //numerid_limits

using spaces::dist_func_t;

/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
template <typename DistType>
struct VecSimIndexAbstract : public VecSimIndexInterface {
protected:
    size_t dim;          // Vector's dimension.
    VecSimType vecType;  // Datatype to index.
    VecSimMetric metric; // Distance metric to use in the index.
    size_t blockSize;    // Index's vector block size (determines by how many vectors to resize when
                         // resizing)
    dist_func_t<DistType>
        dist_func;           // Index's distance function. Chosen by the type, metric and dimension.
    VecSearchMode last_mode; // The last search mode in RediSearch (used for debug/testing).
    bool isMulti;            // Determines if the index should multi-index or not.

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(std::shared_ptr<VecSimAllocator> allocator, size_t dim, VecSimType vecType,
                        VecSimMetric metric, size_t blockSize, bool multi)
        : VecSimIndexInterface(allocator), dim(dim), vecType(vecType), metric(metric),
          blockSize(blockSize ? blockSize : DEFAULT_BLOCK_SIZE), last_mode(EMPTY_MODE),
          isMulti(multi) {
        assert(VecSimType_sizeof(vecType));
        spaces::SetDistFunc(metric, dim, &dist_func);
    }

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexAbstract() {}

    inline dist_func_t<DistType> GetDistFunc() const { return dist_func; }
    inline size_t GetDim() const { return dim; }
    inline void setLastSearchMode(VecSearchMode mode) override { this->last_mode = mode; }
    inline bool isMultiValue() const { return isMulti; }

    template <typename DataType>
    static void NormalizeVector(DataType *input_vector, size_t dim);
};

static const double MAX_FP64 = std::numeric_limits<double>::max();

template <typename DistType>
template <typename DataType>
void VecSimIndexAbstract<DistType>::NormalizeVector(DataType *input_vector, size_t dim) {

    double sum = 0;
    for (size_t i = 0; i < dim; i++) {
        double to_add = (double)input_vector[i] * (double)input_vector[i];
        // Protect from overflow.
        assert(MAX_FP64 - sum >= to_add);
        sum += to_add;
    }

    // Protect fp32 overflow.
    //(if sum is not greater than FP64, then sqrt(sum) is guaranteed to be safe)
    assert(sqrt(sum) <= double(std::numeric_limits<DataType>::max()));

    DataType norm = sqrt(sum);

    for (size_t i = 0; i < dim; i++) {
        input_vector[i] = input_vector[i] / norm;
    }
}
