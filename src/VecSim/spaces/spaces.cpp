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

#define D_R (M_PI / 180.0)

/// @brief Earth's quatratic mean radius for WGS-84
const float EARTH_RADIUS_IN_METERS = 6372797.560856;

static inline double deg_rad(double ang) { return ang * D_R; }
static inline double rad_deg(double ang) { return ang / D_R; }

float GeoDistance2D(const void *p1v, const void *p2v, size_t dummy) {
    float *p1 = (float *)p1v;
    float *p2 = (float *)p2v;
    float lat1r, lon1r, lat2r, lon2r, u, v;
    lat1r = deg_rad(p1[0]);
    lon1r = deg_rad(p1[1]);
    lat2r = deg_rad(p2[0]);
    lon2r = deg_rad(p2[1]);
    u = sin((lat2r - lat1r) / 2);
    v = sin((lon2r - lon1r) / 2);
    return 2.0 * EARTH_RADIUS_IN_METERS * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

float GeoDistance3D(const void *p1v, const void *p2v, size_t dummy) {
    float delta1 = GeoDistance2D(p1v, p2v, dummy);
    float delta2 = ((float *)p1v)[2] - ((float *)p2v)[2];
    return sqrt((delta1 * delta1) + (delta2 * delta2));
}

void SetDistFunc(VecSimMetric metric, size_t dim, dist_func_t<float> *index_dist_func) {

    if (metric == VecSimMetric_Cosine || metric == VecSimMetric_IP) {

        *index_dist_func = IP_FP32_GetDistFunc(dim);

    } else if (metric == VecSimMetric_L2) {

        *index_dist_func = L2_FP32_GetDistFunc(dim);
    } else if (metric == VecSimMetric_WGS84_2D) {
        *index_dist_func = GeoDistance2D;
    } else if (metric == VecSimMetric_WGS84_3D) {
        *index_dist_func = GeoDistance3D;
    }
}

} // namespace spaces
