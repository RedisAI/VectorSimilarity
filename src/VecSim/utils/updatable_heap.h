/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/memory/vecsim_base.h"
#include "vecsim_stl.h"
#include <unordered_map>
#include <map>

namespace vecsim_stl {

// This class implements updatable max heap. insertion, updating and deletion (of the max priority)
// are done in O(log(n)), finding the max priority takes O(1), as well as getting the size and if
// the heap is empty.
// The priority can only be updated DOWN, because we only care about the lowest distance score for a
// vector, and that is the use of this heap. We use it to hold the top candidates while performing
// VSS on multi-valued indexes, and we need to find and delete the worst score easily.
template <typename Priority, typename Value>
class updatable_max_heap : public abstract_priority_queue<Priority, Value> {
private:
    // Maps a priority that exists in the heap to its value.
    std::multimap<Priority, Value, std::greater<Priority>,
                  VecsimSTLAllocator<std::pair<const Priority, Value>>>
        priorityToValue;

    // Maps a value in the heap to its current priority (which is the minimal priority found in the
    // search at the point).
    std::unordered_map<Value, Priority, std::hash<Value>, std::equal_to<Value>,
                       VecsimSTLAllocator<std::pair<const Value, Priority>>>
        valueToPriority;

public:
    updatable_max_heap(const std::shared_ptr<VecSimAllocator> &alloc);
    ~updatable_max_heap();

    inline void emplace(Priority p, Value v) override;
    inline bool empty() const override;
    inline void pop() override;
    inline const std::pair<Priority, Value> top() const override;
    inline size_t size() const override;

private:
    inline auto top_ptr() const;
};

template <typename Priority, typename Value>
updatable_max_heap<Priority, Value>::updatable_max_heap(
    const std::shared_ptr<VecSimAllocator> &alloc)
    : abstract_priority_queue<Priority, Value>(alloc), priorityToValue(alloc),
      valueToPriority(alloc) {}

template <typename Priority, typename Value>
updatable_max_heap<Priority, Value>::~updatable_max_heap() {}

template <typename Priority, typename Value>
size_t updatable_max_heap<Priority, Value>::size() const {
    return valueToPriority.size();
}

template <typename Priority, typename Value>
bool updatable_max_heap<Priority, Value>::empty() const {
    return valueToPriority.empty();
}

template <typename Priority, typename Value>
auto updatable_max_heap<Priority, Value>::top_ptr() const {
    // The `.begin()` of "priorityToValue" is the max priority element.
    auto x = priorityToValue.begin();
    // x has the max priority, but there might be multiple values with the same priority. We need to
    // find the value with the highest value as well.
    auto [begin, end] = priorityToValue.equal_range(x->first);
    auto y = std::max_element(begin, end,
                              [](const auto &a, const auto &b) { return a.second < b.second; });
    return y;
}

template <typename Priority, typename Value>
const std::pair<Priority, Value> updatable_max_heap<Priority, Value>::top() const {
    auto x = top_ptr();
    return *x;
}

template <typename Priority, typename Value>
void updatable_max_heap<Priority, Value>::pop() {
    auto to_remove = top_ptr();
    valueToPriority.erase(to_remove->second);
    priorityToValue.erase(to_remove);
}

template <typename Priority, typename Value>
void updatable_max_heap<Priority, Value>::emplace(Priority p, Value v) {
    // This function either inserting a new value or updating the priority of the value, if the new
    // priority is higher.
    auto existing_v = valueToPriority.find(v);
    if (existing_v == valueToPriority.end()) {
        // Case 1: value is not in the heap. Insert it.
        valueToPriority.emplace(v, p);
        priorityToValue.emplace(p, v);
    } else if (existing_v->second > p) {
        // Case 2: value is in the heap, and its new priority is higher. Update its priority.

        // Because multiple values can get the same priority, we have to find the right node in the
        // `priorityToValue` multimap, otherwise we will delete all entries with this priority.
        // Step 1: find the first entry with the given priority.
        auto pos = priorityToValue.lower_bound(existing_v->second);
        // Step 2: scan mapping to find the right (p, v) pair. We should find it because
        // "existing_v" was found in `valueToPriority`.
        while (pos->second != v) {
            ++pos;
            assert(pos->first == existing_v->second); // We shouldn't get beyond the exact priority.
        }
        priorityToValue.erase(pos);    // Erase by iterator deletes only the specific pair.
        existing_v->second = p;        // Update the priority.
        priorityToValue.emplace(p, v); // Re-insert the updated value to the `priorityToValue` map.
    }
}

} // namespace vecsim_stl
