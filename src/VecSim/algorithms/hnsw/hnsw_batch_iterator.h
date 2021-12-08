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

using CandidatesHeap = vecsim_stl::priority_queue<pair<float, labelType>,
        vecsim_stl::vector<pair<float, labelType>>, CompareByFirst>;

class HNSW_BatchIterator : public VecSimBatchIterator {
private:
    const HNSWIndex *index;
    unsigned int search_id; // the iterator id, used to mark nodes that were returned.
    CandidatesHeap results; // results to return immediately in the next iteration.
    idType entry_point; // internal id of the node to begin the scan from in the next iteration.
    bool allow_marked_candidates; // flag that indicates if we allow the search to visit in nodes that
                                  // where returned in previous iterations
    hnswlib::VisitedList
    CandidatesHeap scanGraph();

public:
    HNSW_BatchIterator(const void *query_vector, const HNSWIndex *index,
                     std::shared_ptr<VecSimAllocator> allocator);

    inline const HNSWIndex *getIndex() const { return index; }

    inline unsigned int getSearchId() const { return search_id; }

    inline bool AllowMarkedCandidates() const { return allow_marked_candidates; }

    inline void setSearchId (unsigned short int id) { search_id = id; }

    inline void setAllowMarkedCandidates() { allow_marked_candidates = true; }

    inline void setEntryPoint(idType node_id) { entry_point = node_id; }

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~HNSW_BatchIterator() override = default;
};
