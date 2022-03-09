#pragma once

#include "VecSim/vec_sim.h"
#include "VecSim/memory/vecsim_base.h"

/**
 * An abstract class for performing search in batches. Every index type should implement its own
 * batch iterator class.
 */
struct VecSimBatchIterator : public VecsimBaseObject {
private:
    void *query_vector_;
    size_t returned_results_count;

public:
    explicit VecSimBatchIterator(const void *query_vector, size_t dim,
                                 std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), returned_results_count(0){
        // Assuming that every element in the vector has 4 bytes.
        query_vector_ = allocator->allocate(dim * 4);
        memcpy(this->query_vector_, query_vector, dim * 4);
    };

    inline const void *getQueryBlob() const { return query_vector_; }

    inline size_t getResultsCount() const { return returned_results_count; }

    inline void updateResultsCount(size_t num) { returned_results_count += num; }

    inline void resetResultsCount() { returned_results_count = 0; }

    // Returns the Top n_res results that *hasn't been returned* in the previous calls.
    // The implementation is specific to the underline index algorithm.
    virtual VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) = 0;

    // Indicates whether there are additional results from the index to return
    virtual bool isDepleted() = 0;

    // Reset the iterator to the initial state, before any results has been returned.
    virtual void reset() = 0;

    virtual ~VecSimBatchIterator() {
        allocator->free_allocation(this->query_vector_);
    };
};
