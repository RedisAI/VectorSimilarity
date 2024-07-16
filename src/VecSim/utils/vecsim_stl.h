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

template <typename Priority, typename Value>
struct abstract_priority_queue : public VecsimBaseObject {
public:
    abstract_priority_queue(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc) {}
    ~abstract_priority_queue() = default;

    virtual inline void emplace(Priority p, Value v) = 0;
    virtual inline bool empty() const = 0;
    virtual inline void pop() = 0;
    virtual inline const std::pair<Priority, Value> top() const = 0;
    virtual inline size_t size() const = 0;
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

    inline void emplace(Priority p, Value v) override { std_queue::emplace(p, v); }
    inline bool empty() const override { return std_queue::empty(); }
    inline void pop() override { std_queue::pop(); }
    inline const std::pair<Priority, Value> top() const override { return std_queue::top(); }
    inline size_t size() const override { return std_queue::size(); }

    // Random order iteration
    inline const auto begin() const { return this->c.begin(); }
    inline const auto end() const { return this->c.end(); }
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

#if defined(__clang__)
#define CLANG_VERSION (__clang_major__ * 100 + __clang_minor__ * 10 + __clang_patchlevel__)
#if (CLANG_VERSION >= 1316) // clang 13.1.6
#define ONE_BYTE_MUTEX_AVAILABLE 1
#endif
#elif (__GNUC__ >= 11)
#define ONE_BYTE_MUTEX_AVAILABLE 1
#else
#define ONE_BYTE_MUTEX_AVAILABLE 0
#endif

#if ONE_BYTE_MUTEX_AVAILABLE != 0
struct one_byte_mutex {
    void lock() {
        if (state.exchange(locked, std::memory_order_acquire) == unlocked)
            return;
        while (state.exchange(sleeper, std::memory_order_acquire) != unlocked)
            state.wait(sleeper, std::memory_order_relaxed);
    }
    void unlock() {
        if (state.exchange(unlocked, std::memory_order_release) == sleeper)
            state.notify_one();
    }

private:
    std::atomic<uint8_t> state{unlocked};

    static constexpr uint8_t unlocked = 0;
    static constexpr uint8_t locked = 0b01;
    static constexpr uint8_t sleeper = 0b10;
};
#endif
} // namespace vecsim_stl
