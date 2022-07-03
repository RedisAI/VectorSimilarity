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
};

// max-heap
template <typename T, typename Container = vecsim_stl::vector<T>,
          typename Compare = std::less<typename Container::value_type>>
using max_priority_queue = std::priority_queue<T, Container, Compare>;

// min-heap
template <typename T, typename Container = vecsim_stl::vector<T>,
          typename Compare = std::greater<typename Container::value_type>>
using min_priority_queue = std::priority_queue<T, Container, Compare>;

template <typename T>
class set : public VecsimBaseObject, public std::set<T, std::less<T>, VecsimSTLAllocator<T>> {
public:
    explicit set(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), std::set<T, std::less<T>, VecsimSTLAllocator<T>>(alloc) {}
};

} // namespace vecsim_stl
