#pragma once
namespace spaces {

template <typename SourceType, typename DestType, typename DistType>
class EncoderInterface {
public:
    virtual void encode(const void *src, void *dest, size_t dim) = 0;
    virtual void decode(const void *src, void *dest, size_t dim) = 0;
    virtual void setDistFunc(VecSimMetric metric, size_t dim,
                             dist_func_t<DistType> *index_dist_func) = 0;
    virtual bool shouldEncode() = 0;
};
} // namespace spaces