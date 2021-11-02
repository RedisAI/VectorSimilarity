#include "VecSim/batch_iterator.h"
#include "VecSim/algorithms/brute_force/brute_force.h"

#include <vector>
#include <limits>

using namespace std;

class BF_BatchIterator : public VecSimBatchIterator {

    const BruteForceIndex *index;
    unsigned char id;
    float lower_bound;
    vector<vector<float>> scores; // vector of scores for every block.
    static unsigned char next_id;

public:
    BF_BatchIterator(const void *query_vector, const BruteForceIndex *index);

    inline const BruteForceIndex *getIndex() const {
        return index;
    };

    VecSimQueryResult_List getNextResults(size_t n_res, VecSimQueryResult_Order order) override;

    bool isDepleted() override;

    void reset() override;

    ~BF_BatchIterator() override = default;
};
