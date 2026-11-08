#pragma once
#include <cstddef>
#include <span>
#include <iosfwd>
#include "VecSim/vec_sim_common.h"

struct QuantizationInit {
    const void *parameters = nullptr;
    bool prepareForTraining = false;
#ifdef BUILD_TESTS
    std::istream *fittedState = nullptr;
#endif
};
struct QuantizationInfo {
    VecSimQuantType type = VecSimQuant_NONE;
    const char *name = nullptr;
    const char *scale = nullptr;
    const char *bias = nullptr;
    size_t vectorBytes = 0;
    size_t queryBytes = 0;
    size_t sharedBytes = 0;
    const char *queryKernel = nullptr;
    const char *graphKernel = nullptr;
};
// Cold-path lifecycle for a storage context. The component wrappers own the context;
// indexes retain a non-owning alias and never delete it separately.
class QuantizationRuntime {
public:
    virtual ~QuantizationRuntime() = default;
    virtual void validateBatch(const void *rows, size_t count, size_t rowBytes,
                               VecSimMetric inputMetric) const = 0;
    // Caller excludes queries and has not stored any vectors. A failed preparation
    // leaves all previous fitted state intact; successful publication keeps dispatch addresses.
    virtual void setMean(std::span<const float> mean) = 0;
    virtual std::span<const float> mean() const = 0;
    virtual size_t storageBytes() const = 0;
    virtual size_t queryBytes() const = 0;
    virtual QuantizationInfo info() const = 0;
#ifdef BUILD_TESTS
    virtual unsigned encodingVersion() const = 0;
    virtual void saveState(std::ostream &output) const = 0;
    virtual void validateStoredRecord(const void *record) const = 0;
#endif
};
