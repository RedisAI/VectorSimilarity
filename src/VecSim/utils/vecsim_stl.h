#pragma once

#include "VecSim/memory/vecsim_malloc.h"
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

template <typename T, typename Container = vecsim_stl::vector<T>,
          typename Compare = std::less<typename Container::value_type>>
using priority_queue = std::priority_queue<T, Container, Compare>;

} // namespace vecsim_stl
