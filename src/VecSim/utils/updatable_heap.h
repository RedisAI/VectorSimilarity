#pragma once

#include "VecSim/memory/vecsim_base.h"
#include "vecsim_stl.h"
#include <unordered_map>
#include <map>

namespace vecsim_stl {

template <typename K, typename V>
class updatable_heap : public VecsimBaseObject {
private:
    std::multimap<V, K, std::greater<V>, VecsimSTLAllocator<std::pair<const V, K>>> scoreToLabel;
    std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                       VecsimSTLAllocator<std::pair<const K, V>>>
        labelToScore;

public:
    updatable_heap(const std::shared_ptr<VecSimAllocator> &alloc);
    ~updatable_heap();

    void emplace(V v, K k);
    bool empty() const;
    void pop();
    const std::pair<V, K> top() const;
    size_t size() const;
};

template <typename K, typename V>
updatable_heap<K, V>::updatable_heap(const std::shared_ptr<VecSimAllocator> &alloc)
    : VecsimBaseObject(alloc), scoreToLabel(alloc), labelToScore(alloc) {}

template <typename K, typename V>
updatable_heap<K, V>::~updatable_heap() {}

template <typename K, typename V>
size_t updatable_heap<K, V>::size() const {
    return labelToScore.size();
}

template <typename K, typename V>
bool updatable_heap<K, V>::empty() const {
    return labelToScore.empty();
}

template <typename K, typename V>
const std::pair<V, K> updatable_heap<K, V>::top() const {
    auto x = scoreToLabel.begin();
    return *x;
}

template <typename K, typename V>
void updatable_heap<K, V>::pop() {
    auto to_remove = scoreToLabel.begin();
    labelToScore.erase(to_remove->second);
    scoreToLabel.erase(to_remove);
}

template <typename K, typename V>
void updatable_heap<K, V>::emplace(V v, K k) {
    auto existing_k = labelToScore.find(k);
    if (existing_k == labelToScore.end()) {
        labelToScore.emplace(k, v);
        scoreToLabel.emplace(v, k);
    } else if (existing_k->second < v) {
        scoreToLabel.erase(existing_k->second);
        existing_k->second = v;
        scoreToLabel.emplace(v, k);
    }
}

} // namespace vecsim_stl
