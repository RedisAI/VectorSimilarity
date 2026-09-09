/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "VecSim/memory/vecsim_base.h"
#include <span>

// Backend-owned training state. Only the serialized tiered writer accesses this interface.
// Notifications describe vectors actually stored in FLAT, after frontend preprocessing.
template <typename DataType>
class QuantizationTrainer : public VecsimBaseObject {
public:
    explicit QuantizationTrainer(std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(std::move(allocator)) {}
    virtual ~QuantizationTrainer() = default;

    virtual void addVector(std::span<const DataType> vector) = 0;
    virtual void removeVector(std::span<const DataType> vector) = 0;
    virtual bool ready() const = 0;

    // The owner excludes backend readers and guarantees an empty graph and no submitted inserts.
    // Implementations must allocate training storage beforehand so committing cannot fail midway.
    virtual void finalize() noexcept = 0;
};
