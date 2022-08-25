#pragma once

#include "VecSim/batch_iterator.h"
#include "hnsw_wrapper.h"
#include "VecSim/spaces/spaces.h"

typedef size_t labelType;
typedef unsigned int idType;

using namespace std;
using spaces::dist_func_t;

using candidatesMinHeap = vecsim_stl::min_priority_queue<float, idType>;
using candidatesMaxHeap = vecsim_stl::max_priority_queue<float, idType>;

class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    HNSWIndex *index_wrapper;
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index;
    dist_func_t<float> dist_func;
    size_t dim;
    hnswlib::VisitedNodesHandler *visited_list; // Pointer to the hnsw visitedList structure.
    unsigned short visited_tag;                 // Used to mark nodes that were scanned.
    idType entry_point;                         // Internal id of the node to begin the scan from.
    bool depleted;
    size_t orig_ef_runtime; // Original default parameter to reproduce.

    // Data structure that holds the search state between iterations.
    float lower_bound;
    candidatesMinHeap top_candidates_extras;
    candidatesMinHeap candidates;

    candidatesMaxHeap scanGraph(candidatesMinHeap &candidates,
                                candidatesMinHeap &spare_top_candidates, float &lower_bound,
                                idType entry_point, VecSimQueryResult_Code *rc);
    VecSimQueryResult_List
    prepareResults(vecsim_stl::max_priority_queue<float, idType> top_candidates, size_t n_res);
    inline void visitNode(idType node_id);
    inline bool hasVisitedNode(idType node_id) const;

public:
    HNSW_BatchIterator(void *query_vector, HNSWIndex *index, VecSimQueryParams *queryParams,
                       std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override;
};
