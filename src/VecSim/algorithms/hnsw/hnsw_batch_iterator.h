#pragma once

#include "VecSim/batch_iterator.h"
#include "hnswlib_c.h"

typedef size_t labelType;
typedef uint idType;

using namespace std;

class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    HNSWIndex *index;
    vecsim_stl::min_priority_queue<pair<float, labelType>>
        results;        // results to return immediately in the next iteration.
    idType entry_point; // internal id of the node to begin the scan from.
    hnswlib::VisitedNodesHandler *visited_list; // Pointer to the hnsw visitedList structure.
    ushort visited_tag; // used to mark nodes that were scanned.
    bool depleted;

    // Save the search state between iterations
    float lower_bound;
    vecsim_stl::max_priority_queue<pair<float, idType>> top_candidates_extras;
    vecsim_stl::min_priority_queue<pair<float, idType>> candidates;

    vecsim_stl::max_priority_queue<pair<float, idType>> scanGraph();
    inline void visitNode(idType node_id);
    inline bool hasVisitedNode(idType node_id) const;

public:
    HNSW_BatchIterator(const void *query_vector, const HNSWIndex *index,
                       std::shared_ptr<VecSimAllocator> allocator, short max_iterations = 500);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override = default;
};
