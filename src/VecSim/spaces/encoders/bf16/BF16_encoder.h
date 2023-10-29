/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once
#include "VecSim/memory/vecsim_base.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/spaces/encoders/encoder.h"

namespace spaces {

using fp32_to_bf16_encoder_t = void (*)(const void *, const void *, size_t);
using bf16_to_fp32_encoder_t = void (*)(const void *, const void *, size_t);

class BF16Encoder : public EncoderInterface<float, bf16, float>, VecsimBaseObject {
protected:
    size_t dim;
    dist_func_t<float> distance_calaculation_func;
    fp32_to_bf16_encoder_t encode_func;
    bf16_to_fp32_encoder_t decode_func;

    fp32_to_bf16_encoder_t setEncodingFunction();
    bf16_to_fp32_encoder_t setDecodingFunction();

    float dist_func(void *encoded, void *not_encoded, size_t dim) {
        float decoded[dim];
        this->decode_func(encoded, decoded, dim);
        return this->distance_calaculation_func(decoded, not_encoded, dim);
    }

public:
    BF16Encoder(std::shared_ptr<VecSimAllocator> allocator, size_t dim)
        : VecsimBaseObject(allocator), dim(dim) {
        this->encode_func = setEncodingFunction();
        this->decode_func = setDecodingFunction();
    }

    virtual void encode(const void *src, void *dest, size_t dim) override {
        this->encode_func(src, dest, dim);
    }
    virtual void decode(const void *src, void *dest, size_t dim) override {}
    virtual void setDistFunc(VecSimMetric metric, size_t dim,
                             dist_func_t<float> *index_dist_func, unsigned char *alignment) override {
        SetDistFunc(metric, dim, index_dist_func, alignment);
    }
    virtual bool shouldEncode() override { return true; }

};

fp32_to_bf16_encoder_t Get_FP32_to_BF16_Encoder(size_t dim, const Arch_Optimization arch_opt, bool big_endian);

bf16_to_fp32_encoder_t Get_BF16_to_FP32_Encoder(size_t dim, const Arch_Optimization arch_opt,
                                                bool big_endian);

} // namespace spaces
