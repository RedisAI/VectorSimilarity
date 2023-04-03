#pragma once
#include <stddef.h>
#include <stdint.h>
#include "vec_sim_common.h"
#include "info_iterator_struct.h"

/**
 * @brief Struct that holds information about the index.
 *
 */
struct VecSimInfo {

    public:
            VecSimAlgo algo;         // Index algorithm.
            size_t indexSize;        // Current count of vectors.
            size_t indexLabelCount;  // Current unique count of labels.
            VecSimMetric metric;     // Index distance metric
            uint64_t memory;         // Index memory consumption.
            VecSimType type;         // Datatype the index holds.
            bool isMulti;            // Determines if the index should multi-index or not.
            size_t dim;              // Vector size (dimension).
            VecSearchMode last_mode; // The mode in which the last query ran.

    virtual VecSimInfoIterator* getIterator();
    virtual ~VecSimInfo() {}

};
