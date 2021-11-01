#include "VecSim/batch_iterator.h"

#include <vector>
#include <limits>

class BF_BatchIterator : public VecSimBatchIterator {

    unsigned char id;
    float lower_bound;
    std::vector<std::pair<float, size_t>> scores;

public:
    BF_BatchIterator(const void *query_vector, const VecSimIndex *index);

    VecSimQueryResult_List getNextResults(size_t n_res) override;

    bool isDepleted() override;

    void reset() override;

    ~BF_BatchIterator() = default;
};
