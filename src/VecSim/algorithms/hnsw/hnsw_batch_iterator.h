#pragma once

#include "VecSim/batch_iterator.h"
#include "hnswlib_c.h"

typedef size_t labelType;
typedef size_t idType;

typedef enum {VISITED_ODD_OFFSET, VISITED_EVEN_OFFSET, RETURNED_VISITED_ODD_OFFSET, RETURNED_VISITED_EVEN_OFFSET} TagOffset;
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
    unique_ptr<CandidatesHeap> results; // results to return immediately in the next iteration.
    idType entry_point; // internal id of the node to begin the scan from in the next iteration.
    bool allow_marked_candidates; // flag that indicates if we allow the search to visit in nodes that
                                  // where returned in previous iterations
    hnswlib::VisitedList *visited_list; // Pointer to the hnsw visitedList structure.
    u_char tag_range_start; // save the minimal tag which is used to mark nodes that were visited and/or
                            // returned by this iterator.
    u_char cur_visited_tag; // used to mark nodes that were scanned in this iteration (that hasn't returned by the iterator).
    u_char cur_returned_visited_tag; // use to mark nodes that were returned in previous iteration,
                                     // and scanned in the current iteration
    bool depleted;

    unique_ptr<CandidatesHeap> scanGraph();
    inline bool hasReturned(idType node_id) const;
    inline void markReturned (uint node_id);
    inline void unmarkReturned (uint node_id);

public:
    HNSW_BatchIterator(const void *query_vector, const HNSWIndex *index,
                     std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override = default;
};
