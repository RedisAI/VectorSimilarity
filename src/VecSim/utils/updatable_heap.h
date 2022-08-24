#pragma once

#include "VecSim/memory/vecsim_base.h"
#include "vecsim_stl.h"
#include <unordered_map>
#include <map>

namespace vecsim_stl {

template <typename P, typename V>
class updatable_max_heap : public priority_queue_abstract<P, V> {
private:
    // Maps a score that exists in the heap to its label.
    std::multimap<P, V, std::greater<P>, VecsimSTLAllocator<std::pair<const P, V>>> scoreToLabel;

    // Maps a label in the heap to its current score (which is the minimal score found in the search
    // at the point).
    std::unordered_map<V, P, std::hash<V>, std::equal_to<V>,
                       VecsimSTLAllocator<std::pair<const V, P>>>
        labelToScore;

public:
    updatable_max_heap(const std::shared_ptr<VecSimAllocator> &alloc);
    ~updatable_max_heap();

    inline void emplace(P p, V v) override;
    inline bool empty() const override;
    inline void pop() override;
    inline const std::pair<P, V> top() const override;
    inline size_t size() const override;
};

template <typename P, typename V>
updatable_max_heap<P, V>::updatable_max_heap(const std::shared_ptr<VecSimAllocator> &alloc)
    : priority_queue_abstract<P, V>(alloc), scoreToLabel(alloc), labelToScore(alloc) {}

template <typename P, typename V>
updatable_max_heap<P, V>::~updatable_max_heap() {}

template <typename P, typename V>
size_t updatable_max_heap<P, V>::size() const {
    return labelToScore.size();
}

template <typename P, typename V>
bool updatable_max_heap<P, V>::empty() const {
    return labelToScore.empty();
}

template <typename P, typename V>
const std::pair<P, V> updatable_max_heap<P, V>::top() const {
    // The `.begin()` of "scoreToLabel" is the max priority element.
    auto x = scoreToLabel.begin();
    return *x;
}

template <typename P, typename V>
void updatable_max_heap<P, V>::pop() {
    auto to_remove = scoreToLabel.begin();
    labelToScore.erase(to_remove->second);
    scoreToLabel.erase(to_remove);
}

template <typename P, typename V>
void updatable_max_heap<P, V>::emplace(P p, V v) {
    // This function either inserting a new value or updating the priority of the value, if the new
    // priority is higher.
    auto existing_v = labelToScore.find(v);
    if (existing_v == labelToScore.end()) {
        // Case 1: value is not in the heap. Insert it.
        labelToScore.emplace(v, p);
        scoreToLabel.emplace(p, v);
    } else if (existing_v->second > p) {
        // Case 2: value is in the heap, and its new priority is higher. Update its priority.

        // Because multiple vectors can get the same score, we have to find the right node in the
        // `scoreToLabel` multimap, otherwise we will delete all entries with this score.
        // Step 1: find the first entry with the given score.
        auto pos = scoreToLabel.lower_bound(existing_v->second);
        // Step 2: scan mapping to find the right (p, v) pair. We should find it because
        // "existing_v" was found in `labelToScore`.
        while (pos->second != v) {
            ++pos;
            assert(pos->first == existing_v->second); // We shouldn't get beyond the exact score.
        }
        scoreToLabel.erase(pos);    // Erase by iterator deletes only the specific pair.
        existing_v->second = p;     // Update the priority.
        scoreToLabel.emplace(p, v); // Re-insert the updated value to the `scoreToLabel` map.
    }
}

} // namespace vecsim_stl
