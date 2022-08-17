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

template <typename K, typename V>
struct priority_queue_abstract : public VecsimBaseObject {
public:
    priority_queue_abstract(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc) {}
    ~priority_queue_abstract() {}

    virtual void emplace(V v, K k) = 0;
    virtual inline bool empty() const = 0;
    virtual inline void pop() = 0;
    virtual inline const std::pair<V, K> top() const = 0;
    virtual inline size_t size() const = 0;
};

// max-heap
// template <typename K, typename V>
// using max_priority_queue = std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V,
// K>>,
//                                                std::less<std::pair<V, K>>>;

// max-heap
template <typename K, typename V>
struct max_priority_queue
    : public priority_queue_abstract<K, V>,
      public std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                                 std::less<std::pair<V, K>>> {
    max_priority_queue(const std::shared_ptr<VecSimAllocator> &alloc)
        : priority_queue_abstract<K, V>(alloc),
          std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                              std::less<std::pair<V, K>>>(alloc) {}
    ~max_priority_queue() {}

    void emplace(V v, K k) override {
        std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                            std::less<std::pair<V, K>>>::emplace(v, k);
    }
    inline bool empty() const override {
        return std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                                   std::less<std::pair<V, K>>>::empty();
    }
    inline void pop() override {
        std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                            std::less<std::pair<V, K>>>::pop();
    }
    inline const std::pair<V, K> top() const override {
        return std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                                   std::less<std::pair<V, K>>>::top();
    }
    inline size_t size() const override {
        return std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                                   std::less<std::pair<V, K>>>::size();
    }
};

// min-heap
template <typename K, typename V>
using min_priority_queue = std::priority_queue<std::pair<V, K>, vecsim_stl::vector<std::pair<V, K>>,
                                               std::greater<std::pair<V, K>>>;

template <typename T>
class set : public VecsimBaseObject, public std::set<T, std::less<T>, VecsimSTLAllocator<T>> {
public:
    explicit set(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::set<T, std::less<T>, VecsimSTLAllocator<T>>(alloc) {}
};

} // namespace vecsim_stl
