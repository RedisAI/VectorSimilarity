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
class updatable_min_max_heap : public abstract_min_max_heap<std::pair<Priority, Value>> {
    using Pair = std::pair<Priority, Value>;

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

    // Temporary storage for a pair to be used in the heap.
    // FIXME: we should return a reference to a pair from `priorityToValue`. This is a workaround
    mutable Pair pair;

public:
    updatable_min_max_heap(const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_min_max_heap<Pair>(alloc), priorityToValue(alloc), valueToPriority(alloc) {}
    ~updatable_min_max_heap() = default;

    inline Pair pop_min() override { return pop<true>(); }
    inline Pair pop_max() override { return pop<false>(); }
    inline size_t size() const override { return valueToPriority.size(); }
    inline bool empty() const override { return valueToPriority.empty(); }
    inline const Pair &peek_min() const override {
        min_ptr();
        return pair;
    }
    inline const Pair &peek_max() const override {
        max_ptr();
        return pair;
    }

    template <typename... Args>
    inline void emplace(Args &&...args) {
        insert(std::make_pair(std::forward<Args>(args)...));
    }

    inline void insert(const Pair &value) override {
        auto [p, v] = value;
        // This function either inserting a new value or updating the priority of the value, if the
        // new priority is higher.
        auto existing_v = valueToPriority.find(v);
        if (existing_v == valueToPriority.end()) {
            // Case 1: value is not in the heap. Insert it.
            valueToPriority.emplace(v, p);
            priorityToValue.emplace(p, v);
        } else if (existing_v->second > p) {
            // Case 2: value is in the heap, and its new priority is higher. Update its priority.

            // Because multiple values can get the same priority, we have to find the right node in
            // the `priorityToValue` multimap, otherwise we will delete all entries with this
            // priority. Step 1: find the first entry with the given priority.
            auto pos = priorityToValue.lower_bound(existing_v->second);
            // Step 2: scan mapping to find the right (p, v) pair. We should find it because
            // "existing_v" was found in `valueToPriority`.
            while (pos->second != v) {
                ++pos;
                assert(pos->first ==
                       existing_v->second); // We shouldn't get beyond the exact priority.
            }
            priorityToValue.erase(pos); // Erase by iterator deletes only the specific pair.
            existing_v->second = p;     // Update the priority.
            priorityToValue.emplace(p,
                                    v); // Re-insert the updated value to the `priorityToValue` map.
        }
    }

    // Random order iteration (actually sorted by priority but not secondarily sorted by value).
    inline auto begin() const { return priorityToValue.begin(); }
    inline auto end() const { return priorityToValue.end(); }

private:
    inline auto min_ptr() const {
        // The `.rbegin()` of "priorityToValue" is the min priority element.
        auto x = priorityToValue.rbegin();
        // x has the min priority, but there might be multiple values with the same priority. We
        // need to find the value with the lowest value as well.
        auto [begin, end] = priorityToValue.equal_range(x->first);
        auto y = std::min_element(begin, end,
                                  [](const auto &a, const auto &b) { return a.second < b.second; });
        pair = std::make_pair(y->first, y->second);
        return y;
    }

    inline auto max_ptr() const {
        // The `.begin()` of "priorityToValue" is the max priority element.
        auto x = priorityToValue.begin();
        // x has the max priority, but there might be multiple values with the same priority. We
        // need to find the value with the highest value as well.
        auto [begin, end] = priorityToValue.equal_range(x->first);
        auto y = std::max_element(begin, end,
                                  [](const auto &a, const auto &b) { return a.second < b.second; });
        pair = std::make_pair(y->first, y->second);
        return y;
    }

    template <bool isMin>
    inline Pair pop() {
        auto to_remove = isMin ? min_ptr() : max_ptr();
        valueToPriority.erase(to_remove->second);
        priorityToValue.erase(to_remove);
        return pair;
    }
};

} // namespace vecsim_stl
