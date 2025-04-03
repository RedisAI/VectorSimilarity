/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/index_factories/components/components_factory.h"

PreprocessorsContainerParams CreatePreprocessorsContainerParams(VecSimMetric metric, size_t dim,
                                                                bool is_normalized,
                                                                unsigned char alignment) {
    // If the index metric is Cosine, and is_normalized == true, we will skip normalizing vectors
    // and query blobs.
    VecSimMetric pp_metric;
    if (is_normalized && metric == VecSimMetric_Cosine) {
        pp_metric = VecSimMetric_IP;
    } else {
        pp_metric = metric;
    }
    return {.metric = pp_metric, .dim = dim, .alignment = alignment};
}
