#pragma once

#include "VecSim/batch_iterator.h"
#include "hnswlib_c.h"

typedef size_t labelType;
typedef uint idType;

using namespace std;

using candidatesMinHeap = vecsim_stl::min_priority_queue<pair<float, idType>>;
using candidatesMaxHeap = vecsim_stl::max_priority_queue<pair<float, idType>>;

class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    HNSWIndex *index_wrapper;
    std::unique_ptr<SpaceInterface<float>> space;
    hnswlib::HierarchicalNSW<float> *hnsw_index;
    hnswlib::VisitedNodesHandler *visited_list; // Pointer to the hnsw visitedList structure.
    unsigned short visited_tag;                         // Used to mark nodes that were scanned.
    idType entry_point;                         // Internal id of the node to begin the scan from.
    bool depleted;

    // Data structure that holds the search state between iterations.
    float lower_bound;
    candidatesMinHeap top_candidates_extras;
    candidatesMinHeap candidates;

    candidatesMaxHeap scanGraph(candidatesMinHeap &candidates,
                                candidatesMinHeap &spare_top_candidates, float &lower_bound,
                                idType entry_point);
    VecSimQueryResult_List
    prepareResults(vecsim_stl::max_priority_queue<pair<float, idType>> top_candidates,
                   size_t n_res);
    inline void visitNode(idType node_id);
    inline bool hasVisitedNode(idType node_id) const;

public:
    HNSW_BatchIterator(const void *query_vector, HNSWIndex *index,
                       std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override = default;
};
