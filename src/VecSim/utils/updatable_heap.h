#pragma once

#include "VecSim/memory/vecsim_base.h"
#include "vecsim_stl.h"
#include <unordered_map>
#include <map>

namespace vecsim_stl {

template <typename K, typename V>
class updatable_max_heap : public priority_queue_abstract<K, V> {
private:
    // Maps a score that exists in the heap to its label.
    std::multimap<V, K, std::greater<V>, VecsimSTLAllocator<std::pair<const V, K>>> scoreToLabel;

    // Maps a label in the heap to its current score (which is the minimal score found in the search
    // at the point).
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       VecsimSTLAllocator<std::pair<const K, V>>>
        labelToScore;

public:
    updatable_max_heap(const std::shared_ptr<VecSimAllocator> &alloc);
    ~updatable_max_heap();

    void emplace(V v, K k) override;
    inline bool empty() const override;
    inline void pop() override;
    inline const std::pair<V, K> top() const override;
    inline size_t size() const override;
};

template <typename K, typename V>
updatable_max_heap<K, V>::updatable_max_heap(const std::shared_ptr<VecSimAllocator> &alloc)
    : priority_queue_abstract<K, V>(alloc), scoreToLabel(alloc), labelToScore(alloc) {}

template <typename K, typename V>
updatable_max_heap<K, V>::~updatable_max_heap() {}

template <typename K, typename V>
size_t updatable_max_heap<K, V>::size() const {
    return labelToScore.size();
}

template <typename K, typename V>
bool updatable_max_heap<K, V>::empty() const {
    return labelToScore.empty();
}

template <typename K, typename V>
const std::pair<V, K> updatable_max_heap<K, V>::top() const {
    // The `.begin()` of "scoreToLabel" is the max priority element.
    auto x = scoreToLabel.begin();
    return *x;
}

template <typename K, typename V>
void updatable_max_heap<K, V>::pop() {
    auto to_remove = scoreToLabel.begin();
    labelToScore.erase(to_remove->second);
    scoreToLabel.erase(to_remove);
}

template <typename K, typename V>
void updatable_max_heap<K, V>::emplace(V v, K k) {
    // This function either inserting a new value or updating the priority of the key, if the new
    // priority is higher.
    auto existing_k = labelToScore.find(k);
    if (existing_k == labelToScore.end()) {
        // Case 1: key is not in the heap. Insert it.
        labelToScore.emplace(k, v);
        scoreToLabel.emplace(v, k);
    } else if (existing_k->second > v) {
        // Case 2: key is in the heap, and its new priority is higher. Update its priority.

        // Because multiple vectors can get the same score, we have to find the right node in the
        // `scoreToLabel` multimap, otherwise we will delete all entries with this score.
        // Step 1: find the first entry with the given score.
        auto pos = scoreToLabel.lower_bound(existing_k->second);
        // Step 2: scan mapping to find the right (v, k) pair. We should find it because
        // "existing_k" was found in `labelToScore`.
        while (pos->second != k) {
            ++pos;
            assert(pos->first == existing_k->second); // We shouldn't get beyond the exact score.
        }
        scoreToLabel.erase(pos);    // Erase by iterator deletes only the specific pair.
        existing_k->second = v;     // Update the priority.
        scoreToLabel.emplace(v, k); // Re-insert the updated value to the `scoreToLabel` map.
    }
}

} // namespace vecsim_stl
