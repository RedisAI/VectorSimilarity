#pragma once
#include "vec_sim_common.h"
#include "vec_sim_interface.h"
#include "query_results.h"
#include <stddef.h>
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/space_interface.h"
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
    DISTFUNC<float> dist_func;
    VecSearchMode last_mode;
    std::shared_ptr<SpaceInterface<float>> space;

public:
    /**
     * @brief Construct a new Vec Sim Index object
     *
     */
    VecSimIndexAbstract(std::shared_ptr<VecSimAllocator> allocator, size_t dim, VecSimType vecType,
                        VecSimMetric metric, size_t blockSize, SpaceInterface<float> *space)
        : VecSimIndexInterface(allocator), dim(dim), vecType(vecType), metric(metric),
          blockSize(blockSize ? blockSize : DEFAULT_BLOCK_SIZE), last_mode(EMPTY_MODE),
          space(space) {
        dist_func = space->get_dist_func();
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
