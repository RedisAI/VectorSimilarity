#include "VecSim/algorithms/brute_force/brute_force_factory.h"
#include "VecSim/algorithms/brute_force/brute_force.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"

VecSimIndex *BruteForceFactory::NewIndex(const BFParams *params,
                                         std::shared_ptr<VecSimAllocator> allocator) {
    // check if single and return new bf_index
    assert(!params->multi);
    return new (allocator) BruteForceIndex_Single(params, allocator);
}
/* static size_t EstimateInitialSize(const BFParams *params);
 static size_t EstimateElementSize(const BFParams *params); */
