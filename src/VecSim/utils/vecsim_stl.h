#pragma once

#include "VecSim/memory/vecsim_base.h"
#include <vector>
#include <set>
#include <unordered_map>
#include <queue>

namespace vecsim_stl {

template <typename T>
using vector = std::vector<T, VecsimSTLAllocator<T>>;

template <typename T>
using set = std::set<T, std::less<T>, VecsimSTLAllocator<T>>;

template <typename K, typename V>
using unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                                         VecsimSTLAllocator<std::pair<const K, V>>>;

// max-heap
template <typename T, typename Container = vecsim_stl::vector<T>,
          typename Compare = std::less<typename Container::value_type>>
using max_priority_queue = std::priority_queue<T, Container, Compare>;

// min-heap
template <typename T, typename Container = vecsim_stl::vector<T>,
          typename Compare = std::greater<typename Container::value_type>>
using min_priority_queue = std::priority_queue<T, Container, Compare>;

template <typename T>
class set_wrapper : public VecsimBaseObject {
public:
    vecsim_stl::set<T> set_;
    explicit set_wrapper(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), set_(alloc) {}
};

} // namespace vecsim_stl
