#include "VecSim/spaces/space_aux.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/IP_space.h"
#include "VecSim/spaces/L2_space.h"
namespace spaces {

/*** Defined in vec_sim_common.h
    NO_OPTIMIZATION = 0,
    Ext16 = 1,          // dim % 16 == 0
    Ext4 = 2,           // dim % 4 == 0
    ExtResiduals16 = 3, // dim > 16 && dim % 16 < 4
    ExtResiduals4 = 4,  // dim > 4
};
***/
CalculationGuideline GetCalculationGuideline(size_t dim) {

    CalculationGuideline ret_score = NO_OPTIMIZATION;

    if (dim % 16 == 0) {
        ret_score = Ext16;
    } else if (dim % 4 == 0) {
        ret_score = Ext4;
    } else if (dim > 16 && dim % 16 < 4) {
        ret_score = ExtResiduals16;
    } else if (dim > 4) {
        ret_score = ExtResiduals4;
    }
    return ret_score;
}

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func) {

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {

        *index_dist_func = IP_FP32_GetOptDistFunc(dim);

    } else if (metric == VecSimMetric_L2) {

        *index_dist_func = L2_FP32_GetOptDistFunc(dim);
    }
}

} // namespace spaces
