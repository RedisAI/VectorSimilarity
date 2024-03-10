#pragma once
#include "encoder.h"
#include "VecSim/spaces/spaces.h"
namespace spaces {

template <typename SourceType, typename DestType, typename DistType>
class NoOpEncoder : public EncoderInterface<SourceType, DestType, DistType> {
public:
    virtual void encode(const void *src, void *dest, size_t dim) override {}
    virtual void decode(const void *src, void *dest, size_t dim) override {}
    virtual void setDistFunc(VecSimMetric metric, size_t dim,
                             dist_func_t<DistType> *index_dist_func) override {
        spaces::SetDistFunc(metric, dim, index_dist_func);
    }
    virtual bool shouldEncode() override { return false; }
};
} // namespace spaces