#include "batch_iterator.h"

#include <vector>

class BF_BatchIterator : public VecSimBatchIterator {

    unsigned char id;
    float lower_bound;
    std::vector<std::pair<float, size_t>> scores;

public:
    BF_BatchIterator(const void *query_vector, const VecSimIndex *index) :
            VecSimBatchIterator(query_vector, index) {
        id = 0;
        lower_bound = -INF;
    }

    VecSimQueryResult_List getNextResults(size_t n_res) override;

    bool isDepleted() override;

    void reset() override;

    ~BF_BatchIterator() = default;
};
