#pragma once

#include "VecSim/memory/vecsim_base.h"
#include <vector>
#include <set>
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

template <typename P, typename V>
struct priority_queue_abstract : public VecsimBaseObject {
public:
    priority_queue_abstract(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc) {}
    ~priority_queue_abstract() {}

    virtual void emplace(P p, V v) = 0;
    virtual inline bool empty() const = 0;
    virtual inline void pop() = 0;
    virtual inline const std::pair<P, V> top() const = 0;
    virtual inline size_t size() const = 0;
};

// max-heap
template <typename P, typename V>
struct max_priority_queue : public priority_queue_abstract<P, V> {
private:
    std::priority_queue<std::pair<P, V>, vecsim_stl::vector<std::pair<P, V>>,
                        std::less<std::pair<P, V>>>
        max_pq;

public:
    max_priority_queue(const std::shared_ptr<VecSimAllocator> &alloc)
        : priority_queue_abstract<P, V>(alloc), max_pq(alloc) {}
    ~max_priority_queue() {}

    void emplace(P p, V v) override { max_pq.emplace(p, v); }
    inline bool empty() const override { return max_pq.empty(); }
    inline void pop() override { max_pq.pop(); }
    inline const std::pair<P, V> top() const override { return max_pq.top(); }
    inline size_t size() const override { return max_pq.size(); }
};

// min-heap
template <typename P, typename V>
using min_priority_queue = std::priority_queue<std::pair<P, V>, vecsim_stl::vector<std::pair<P, V>>,
                                               std::greater<std::pair<P, V>>>;

template <typename T>
class set : public VecsimBaseObject, public std::set<T, std::less<T>, VecsimSTLAllocator<T>> {
public:
    explicit set(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::set<T, std::less<T>, VecsimSTLAllocator<T>>(alloc) {}
};

} // namespace vecsim_stl
