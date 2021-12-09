#pragma once

#include "VecSim/batch_iterator.h"
#include "hnswlib_c.h"

typedef size_t labelType;
typedef size_t idType;

// use the maximum uint as the invalid id, since the search id is represented by unsigned short
#define INVALID_SEARCH_ID std::numeric_limits<unsigned int>::max()

using namespace std;

struct CompareByFirst {
    constexpr bool operator()(pair<float, labelType> const &a,
                              pair<float, labelType> const &b) const noexcept {
        return a.first < b.first;
    }
};

using CandidatesHeap = vecsim_stl::priority_queue<pair<float, labelType>>;

class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    const HNSWIndex *index;
    CandidatesHeap results; // results to return immediately in the next iteration.
    idType entry_point; // internal id of the node to begin the scan from in the next iteration.
    bool allow_marked_candidates; // flag that indicates if we allow the search to visit in nodes that
                                  // where returned in previous iterations
    hnswlib::VisitedList *visited_list; // Pointer to the hnsw visitedList structure.
    uint minimal_tag; // save the minimal tag which is used for nodes that were visited by this iterator.
    uint visited_tag; // use to mark nodes that were scanned in this iteration.
    uint visited_and_returned_tag; // use to mark nodes that were returned in previous iteration,
                                   // and scanned in the current iteration
    bool depleted;

    CandidatesHeap scanGraph();
    bool hasReturned(idType node_id) const;

public:
    HNSW_BatchIterator(const void *query_vector, const HNSWIndex *index,
                     std::shared_ptr<VecSimAllocator> allocator);

    inline void markReturned (uint node_id) {
        this->visited_list->visitedElements[node_id] = this->visited_and_returned_tag;
    }

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override = default;
};
