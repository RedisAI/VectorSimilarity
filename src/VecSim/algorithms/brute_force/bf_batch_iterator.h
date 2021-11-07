#include "VecSim/batch_iterator.h"
#include "VecSim/algorithms/brute_force/brute_force.h"

#include <vector>
#include <limits>

using namespace std;

class BF_BatchIterator : public VecSimBatchIterator {

    const BruteForceIndex *index;
    unsigned char id;
    vector<vector<pair<float, labelType>>>
        scores; // vector of scores for every block ("score matrix").
    unordered_map<labelType, pair<size_t, size_t>> labelToScoreCoordinates;
    static unsigned char next_id;

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
