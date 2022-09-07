#pragma once

#include "VecSim/batch_iterator.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/vec_sim_common.h" //labelType, idType
#include "VecSim/query_result_struct.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/algorithms/hnsw/visited_nodes_handler.h"

using spaces::dist_func_t;

template <typename DataType, typename DistType>
class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    HNSWIndex<DataType, DistType> *index;
    dist_func_t<DistType> dist_func;
    size_t dim;
    VisitedNodesHandler *visited_list; // Pointer to the hnsw visitedList structure.
    tag_t visited_tag;                 // Used to mark nodes that were scanned.
    idType entry_point;                // Internal id of the node to begin the scan from.
    bool depleted;
    size_t orig_ef_runtime; // Original default parameter to reproduce.

    // Data structure that holds the search state between iterations.
    using candidatesMinHeap = vecsim_stl::min_priority_queue<DistType, idType>;
    using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
    DistType lower_bound;
    candidatesMinHeap top_candidates_extras;
    candidatesMinHeap candidates;

    candidatesMaxHeap scanGraph(candidatesMinHeap &candidates,
                                candidatesMinHeap &spare_top_candidates, DistType &lower_bound,
                                idType entry_point, VecSimQueryResult_Code *rc);
    VecSimQueryResult_List prepareResults(candidatesMaxHeap top_candidates, size_t n_res);
    inline void visitNode(idType node_id) {
        this->visited_list->tagNode(node_id, this->visited_tag);
    }
    inline bool hasVisitedNode(idType node_id) const {
        return this->visited_list->getNodeTag(node_id) == this->visited_tag;
    }

public:
    HNSW_BatchIterator(void *query_vector, HNSWIndex<DataType, DistType> *index,
                       VecSimQueryParams *queryParams, std::shared_ptr<VecSimAllocator> allocator);

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override;
};
