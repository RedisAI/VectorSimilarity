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
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/vec_sim_common.h"

#include <cassert>
#include <mutex>

/**
 * A set of internal ids, backed by one bit per id in the index. insert() and contains() are O(1),
 * and clear() resets only the bits that were actually set, so the same set can serve call after
 * call without a zero fill proportional to the index capacity.
 *
 * The delete and repair paths need such a set per call, but never put more than a handful of ids
 * in it (a node's links, and its neighbors' links). Building a fresh bitmap per call costs an
 * allocation plus a zero fill over the whole index; reusing a pooled set costs neither.
 */
class IdFlagSet : public VecsimBaseObject {
public:
    IdFlagSet(size_t capacity, const std::shared_ptr<VecSimAllocator> &allocator)
        : VecsimBaseObject(allocator), flags(capacity, false, allocator), set_ids(allocator) {}

    void insert(idType id) {
        assert(id < flags.size());
        if (!flags[id]) {
            flags[id] = true;
            set_ids.push_back(id);
        }
    }

    bool contains(idType id) const {
        assert(id < flags.size());
        return flags[id];
    }

    void clear() {
        for (idType id : set_ids) {
            flags[id] = false;
        }
        set_ids.clear();
    }

    // Assumes the set is empty, so that dropping the flags cannot lose a set bit.
    void resize(size_t capacity) {
        assert(set_ids.empty() && "an id set must be cleared before it is resized");
        flags.resize(capacity, false);
        flags.shrink_to_fit();
        set_ids.shrink_to_fit();
    }

private:
    vecsim_stl::vector<bool> flags;
    // The ids whose bit is currently set, so that clear() is proportional to the set's size and
    // not to the index capacity.
    vecsim_stl::vector<idType> set_ids;
};

/**
 * The two id sets that one delete or one repair job needs, handed out together so that a caller
 * takes and returns them in a single pool round-trip.
 */
class IdFlagSetPair : public VecsimBaseObject {
public:
    IdFlagSetPair(size_t capacity, const std::shared_ptr<VecSimAllocator> &allocator)
        : VecsimBaseObject(allocator), first(capacity, allocator), second(capacity, allocator) {}

    void clear() {
        first.clear();
        second.clear();
    }

    void resize(size_t capacity) {
        first.resize(capacity);
        second.resize(capacity);
    }

    IdFlagSet first;
    IdFlagSet second;
};

/**
 * A pool of id set pairs, so that concurrent repair jobs each get their own sets without
 * allocating them per call. Mirrors VisitedNodesHandlerPool, which plays the same role for graph
 * scans.
 */
class IdFlagSetPool : public VecsimBaseObject {
public:
    IdFlagSetPool(size_t capacity, const std::shared_ptr<VecSimAllocator> &allocator)
        : VecsimBaseObject(allocator), pool(allocator), capacity(capacity), sets_in_use(0) {}

    IdFlagSetPair *get() {
        std::unique_lock<std::mutex> lock(pool_guard);
        if (pool.empty()) {
            sets_in_use++;
            return new (this->allocator) IdFlagSetPair(capacity, this->allocator);
        }
        IdFlagSetPair *set = pool.back();
        pool.pop_back();
        return set;
    }

    // Takes a set back, cleared and ready for the next caller.
    void put(IdFlagSetPair *set) {
        set->clear();
        std::unique_lock<std::mutex> lock(pool_guard);
        pool.push_back(set);
    }

    // This should be called under a guarded section only (NOT in parallel), like the equivalent
    // VisitedNodesHandlerPool::resize.
    void resize(size_t new_capacity) {
        assert(sets_in_use == pool.size()); // validate that no set is in use outside the pool.
        capacity = new_capacity;
        if (new_capacity == 0) {
            // The index holds no elements, so hand the scratch memory back rather than keeping
            // empty sets around.
            clearPool();
            return;
        }
        for (auto *set : pool) {
            set->resize(new_capacity);
        }
    }

    void clearPool() {
        for (auto *set : pool) {
            delete set;
        }
        pool.clear();
        pool.shrink_to_fit();
        sets_in_use = 0;
    }

    size_t getPoolSize() const { return pool.size(); }

    ~IdFlagSetPool() override { clearPool(); }

private:
    vecsim_stl::vector<IdFlagSetPair *> pool;
    std::mutex pool_guard;
    size_t capacity;
    size_t sets_in_use;
};

/**
 * Takes a pair of sets from the pool and returns them, cleared, on scope exit, so that an early
 * return or an exception cannot leak them out of the pool.
 */
class PooledIdFlagSets {
public:
    explicit PooledIdFlagSets(IdFlagSetPool &pool) : pool(pool), sets(pool.get()) {}
    ~PooledIdFlagSets() { pool.put(sets); }

    PooledIdFlagSets(const PooledIdFlagSets &) = delete;
    PooledIdFlagSets &operator=(const PooledIdFlagSets &) = delete;

    IdFlagSet &first() const { return sets->first; }
    IdFlagSet &second() const { return sets->second; }

private:
    IdFlagSetPool &pool;
    IdFlagSetPair *sets;
};
