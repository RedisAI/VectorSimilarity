#pragma once

#include "VecSim/memory/vecsim_base.h"
#include <utility>
#include <vector>
#include <set>
#include <unordered_map>
#include <queue>
#include <src/bm.h>

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
class set_wrapper : public VecsimBaseObject, public vecsim_stl::set<T> {
public:
    explicit set_wrapper(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc), vecsim_stl::set<T>(alloc) {}
};

class dbg_block_allocator
{
	std::shared_ptr<VecSimAllocator> baseAllocator;
public:
	explicit dbg_block_allocator(std::shared_ptr<VecSimAllocator> baseAllocator_) : baseAllocator(std::move(baseAllocator_)) {}
	bm::word_t* allocate(size_t n, const void *)
	{
		return (bm::word_t*) baseAllocator->allocate(n * sizeof(bm::word_t));
	}

	void deallocate(bm::word_t* p, size_t /* n */)
	{
		baseAllocator->deallocate(p, 0);
	}
};

typedef bm::mem_alloc<dbg_block_allocator, VecSimAllocator,
bm::alloc_pool<dbg_block_allocator, VecSimAllocator>> dbg_alloc;

template <typename Alloc>
class bvector : public VecsimBaseObject, public bm::bvector<Alloc>{
public:
	explicit bvector(const std::shared_ptr<VecSimAllocator> &alloc)
			: VecsimBaseObject(alloc),
			  bm::bvector<Alloc> (bm::BM_BIT, bm::gap_len_table<true>::_len,
			                      bm::id_max,
								  vecsim_stl::dbg_alloc(vecsim_stl::dbg_block_allocator(alloc),
											VecSimAllocator(alloc))) {}
};

} // namespace vecsim_stl
