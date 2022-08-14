#pragma once
#include "vec_sim_common.h"
#include "vec_sim_interface.h"
#include "query_results.h"
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "info_iterator_struct.h"

/**
 * @brief Abstract C++ class for vector index, delete and lookup
 *
 */
struct VecSimIndexAbstract : public VecSimIndexInterface {
protected:
    size_t dim;
    VecSimType vecType;
    VecSimMetric metric;
    size_t blockSize;
    Spaces::dist_func_t<float> dist_func;
    VecSearchMode last_mode;

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(std::shared_ptr<VecSimAllocator> allocator, size_t dim, VecSimType vecType,
                        VecSimMetric metric, size_t blockSize)
        : VecSimIndexInterface(allocator), dim(dim), vecType(vecType), metric(metric),
          blockSize(blockSize ? blockSize : DEFAULT_BLOCK_SIZE), last_mode(EMPTY_MODE) {
        assert(VecSim_SizeOfType(vecType));
        Spaces::SetDistFunc(metric, dim, &dist_func);
    }

    /**
     * @brief Destroy the Vec Sim Index object
     *
     */
    virtual ~VecSimIndexAbstract() {}

public:
    static timeoutCallbackFunction timeoutCallback;

    static void setTimeoutCallbackFunction(timeoutCallbackFunction callback);
};
