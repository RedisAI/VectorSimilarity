#pragma once

#include "VecSim/memory/vecsim_base.h"
#include "vecsim_stl.h"
#include <unordered_map>
#include <map>

namespace vecsim_stl {

template <typename K, typename V>
class updatable_max_heap : public priority_queue_abstract<K, V> {
private:
    std::multimap<V, K, std::greater<V>, VecsimSTLAllocator<std::pair<const V, K>>> scoreToLabel;
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
    auto existing_k = labelToScore.find(k);
    if (existing_k == labelToScore.end()) {
        labelToScore.emplace(k, v);
        scoreToLabel.emplace(v, k);
    } else if (existing_k->second > v) {
        auto pos = scoreToLabel.lower_bound(existing_k->second);
        while (pos->second != k) {
            ++pos;
        }
        scoreToLabel.erase(pos);
        existing_k->second = v;
        scoreToLabel.emplace(v, k);
    }
}

} // namespace vecsim_stl
