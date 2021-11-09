#include "VecSim/batch_iterator.h"
#include "VecSim/algorithms/brute_force/brute_force.h"

#include <vector>
#include <limits>

using namespace std;

class BF_BatchIterator : public VecSimBatchIterator {
private:
    const BruteForceIndex *index;
    unsigned char id;
    vector<pair<float, labelType>> scores; // vector of scores for every label.
    static unsigned char next_id; // this holds the next available id to be used by a new instance.

    VecSimQueryResult *searchByHeuristics(size_t n_res, VecSimQueryResult_Order order);
    VecSimQueryResult *selectBasedSearch(size_t n_res);
    VecSimQueryResult *heapBasedSearch(size_t n_res);

public:
    BF_BatchIterator(const void *query_vector, const BruteForceIndex *index);

    inline const BruteForceIndex *getIndex() const { return index; };

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~BF_BatchIterator() override = default;
};
