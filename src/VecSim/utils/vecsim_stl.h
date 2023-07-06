/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/memory/vecsim_base.h"
#include <vector>
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
};

template <typename T>
struct abstract_min_max_heap : public VecsimBaseObject {
public:
    abstract_min_max_heap(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc) {}
    ~abstract_min_max_heap() {}

    virtual inline void insert(const T &value) = 0;
    virtual inline size_t size() const = 0;
    virtual inline bool empty() const = 0;
    virtual inline T pop_min() = 0;
    virtual inline T pop_max() = 0;
    virtual inline const T &peek_min() const = 0;
    virtual inline const T &peek_max() const = 0;
    virtual inline T exchange_min(const T &value) = 0; // combines pop-and-then-insert logic
    virtual inline T exchange_max(const T &value) = 0; // combines pop-and-then-insert logic

    // convenience methods
    template <typename... Args>
    inline T exchange_max(Args &&...args) {
        return exchange_max(static_cast<const T &>(T(args...)));
    }
    template <typename... Args>
    inline T exchange_min(Args &&...args) {
        return exchange_min(static_cast<const T &>(T(args...)));
    }
    template <typename... Args>
    inline void emplace(Args &&...args) {
        insert(T(std::forward<Args>(args)...));
    }
};

// max-heap
template <typename Priority, typename Value>
using max_priority_queue =
    std::priority_queue<std::pair<Priority, Value>, vecsim_stl::vector<std::pair<Priority, Value>>,
                        std::less<std::pair<Priority, Value>>>;

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
