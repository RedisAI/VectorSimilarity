#pragma once

#include "VecSim/vec_sim.h"
// For VecSimIndexAbstract
#include "VecSim/vec_sim_index.h"
// For labelType
#include "VecSim/vec_sim_common.h"

// Non-CUDA Interface of the RaftIVF index to avoid importing CUDA code
// in the tiered index.
template <typename DataType, typename DistType = DataType>
struct RaftIvfInterface : public VecSimIndexAbstract<DistType>
{
    RaftIvfInterface(const AbstractIndexInitParams &params) : VecSimIndexAbstract<DistType>(params) {}
    virtual uint32_t nLists() const = 0;
    virtual inline void setNProbes(uint32_t n_probes) = 0;

    virtual int addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                               void *auxiliaryCtx = nullptr) = 0;
};
