#include "VecSim/algorithms/brute_force/brute_force_factory.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/vec_sim_common.h" // labelType

namespace BruteForceFactory {
VecSimIndex *NewIndex(const BFParams *params, std::shared_ptr<VecSimAllocator> allocator) {
    // check if single and return new bf_index
    assert(!params->multi);
    return new (allocator) BruteForceIndex_Single<float, float>(params, allocator);
}

size_t EstimateInitialSize(const BFParams *params) {

    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + sizeof(size_t);
    if (!params->multi)
        est += sizeof(BruteForceIndex_Single<float, float>);

    // Parameters related part.

    if (params->initialCapacity) {
        est += params->initialCapacity * sizeof(labelType) + sizeof(size_t);
    }

    return est;
}

size_t EstimateElementSize(const BFParams *params) {
    return params->dim * VecSimType_sizeof(params->type) + sizeof(labelType);
}
}; // namespace BruteForceFactory
