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
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>

namespace vecsim_stl {

template <typename K, typename V>
using unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                                         VecsimSTLAllocator<std::pair<const K, V>>>;

template <typename T>
class vector : public VecsimBaseObject, public std::vector<T, VecsimSTLAllocator<T>> {
public:
    explicit vector(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::vector<T, VecsimSTLAllocator<T>>(alloc) {}
    explicit vector(size_t cap, const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::vector<T, VecsimSTLAllocator<T>>(cap, alloc) {}
    explicit vector(size_t cap, T val, const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::vector<T, VecsimSTLAllocator<T>>(cap, val, alloc) {}

    bool remove(T element) {
        auto it = std::find(this->begin(), this->end(), element);
        if (it != this->end()) {
            // Swap the last element with the current one (equivalent to removing the element from
            // the list).
            *it = this->back();
            this->pop_back();
            return true;
        }
        return false;
    }
};

template <typename Priority, typename Value>
struct abstract_priority_queue : public VecsimBaseObject {
public:
    abstract_priority_queue(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc) {}
    ~abstract_priority_queue() = default;

    virtual void emplace(Priority p, Value v) = 0;
    virtual bool empty() const = 0;
    virtual void pop() = 0;
    virtual const std::pair<Priority, Value> top() const = 0;
    virtual size_t size() const = 0;
};

// max-heap
template <typename Priority, typename Value,
          typename std_queue = std::priority_queue<std::pair<Priority, Value>,
                                                   vecsim_stl::vector<std::pair<Priority, Value>>,
                                                   std::less<std::pair<Priority, Value>>>>
struct max_priority_queue : public abstract_priority_queue<Priority, Value>, public std_queue {
public:
    max_priority_queue(const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_priority_queue<Priority, Value>(alloc), std_queue(alloc) {}
    ~max_priority_queue() = default;

    void emplace(Priority p, Value v) override { std_queue::emplace(p, v); }
    bool empty() const override { return std_queue::empty(); }
    void pop() override { std_queue::pop(); }
    const std::pair<Priority, Value> top() const override { return std_queue::top(); }
    size_t size() const override { return std_queue::size(); }

    // Random order iteration
    const auto begin() const { return this->c.begin(); }
    const auto end() const { return this->c.end(); }
};

// min-heap
template <typename Priority, typename Value>
using min_priority_queue =
    std::priority_queue<std::pair<Priority, Value>, vecsim_stl::vector<std::pair<Priority, Value>>,
                        std::greater<std::pair<Priority, Value>>>;

template <typename T>
class set : public VecsimBaseObject, public std::set<T, std::less<T>, VecsimSTLAllocator<T>> {
public:
    explicit set(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::set<T, std::less<T>, VecsimSTLAllocator<T>>(alloc) {}
};

template <typename T>
class unordered_set
    : public VecsimBaseObject,
      public std::unordered_set<T, std::hash<T>, std::equal_to<T>, VecsimSTLAllocator<T>> {
public:
    explicit unordered_set(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc),
          std::unordered_set<T, std::hash<T>, std::equal_to<T>, VecsimSTLAllocator<T>>(alloc) {}
    explicit unordered_set(size_t n_bucket, const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc),
          std::unordered_set<T, std::hash<T>, std::equal_to<T>, VecsimSTLAllocator<T>>(n_bucket,
                                                                                       alloc) {}
};

} // namespace vecsim_stl
