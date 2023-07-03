/*
 * Copyright Redis Ltd. 2016 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */
#pragma once

#include <bit>
#include <assert.h>
#include "VecSim/memory/vecsim_base.h"
#include "vecsim_stl.h"

namespace vecsim_stl {

template <typename T, typename Compare = std::less<T>>
class min_max_heap : public VecsimBaseObject {
private:
    vecsim_stl::vector<T> data;

private:
    inline bool is_min(size_t index) const;
    inline size_t parent(size_t index) const;
    inline size_t left_child(size_t index) const;
    inline size_t right_child(size_t index) const;
    inline void swap(size_t index1, size_t index2);

    inline void bubble_up(size_t index);
    template <bool min>
    void bubble_up_kind(size_t index);

    template <bool min>
    void trickle_down(size_t index);

    inline bool compare(size_t index1, size_t index2) const {
        assert(index1 < data.size() && index2 < data.size());
        assert(index1 > 0 && index2 > 0);
        assert(index1 != index2);
        return Compare()(data[index1], data[index2]);
    }

    // trickle down (min and max) helpers
    inline char highest_descendant_in_range(size_t index) const;
    template <bool min>
    inline size_t choose_from_2(size_t a, size_t b) const {
        return (min ? compare(a, b) : compare(b, a)) ? a : b;
    }
    template <bool min>
    inline size_t choose_from_3(size_t a, size_t b, size_t c) const {
        return (min ? compare(a, b) : compare(b, a)) ? choose_from_2<min>(a, c)
                                                         : choose_from_2<min>(b, c);
    }
    template <bool min>
    inline size_t choose_from_4(size_t a, size_t b, size_t c, size_t d) const {
        return (min ? compare(a, b) : compare(b, a)) ? choose_from_3<min>(a, c, d)
                                                         : choose_from_3<min>(b, c, d);
    }
    template <bool min>
    inline size_t index_best_child_grandchild(size_t index) const;

public:
    min_max_heap(const std::shared_ptr<VecSimAllocator> &alloc);
    min_max_heap(size_t size, const std::shared_ptr<VecSimAllocator> &alloc);
    ~min_max_heap() = default;

    inline size_t size() const { return data.size() - 1; }
    inline bool empty() const { return size() == 0; }
    inline void clear() {
        data.clear();
        data.push_back(T()); // dummy element
    }

    inline void insert(const T &value);
    inline T pop_min();
    inline T pop_max();
    inline const T &peek_min() const;
    inline const T &peek_max() const;
    inline T exchange_min(const T &value); // combines pop-and-then-insert logic
    inline T exchange_max(const T &value); // combines pop-and-then-insert logic
};

/********************************** Ctor / Dtor **********************************/

template <typename T, typename Compare>
min_max_heap<T, Compare>::min_max_heap(const std::shared_ptr<VecSimAllocator> &alloc)
    : VecsimBaseObject(alloc), data(alloc) {
    data.push_back(T()); // dummy element
}

template <typename T, typename Compare>
min_max_heap<T, Compare>::min_max_heap(size_t size, const std::shared_ptr<VecSimAllocator> &alloc)
    : VecsimBaseObject(alloc), data(alloc) {
    data.reserve(size + 1);
    data.push_back(T()); // dummy element
}

/**************************** Private Implementation *****************************/

/*
 * `is_min` returns true if the index is a min node, false otherwise.
 * A node is a min node if its level (depth) is odd (and the root has a depth 1).
 * With our array representation, a node is a min node if the log2 floor of its index is even.
 * (log2 floor of 1 is 0 - min, log2 floor of 2 is 1 - max, log2 floor of 3 is 1 - max, log2 floor
 * of 4 is 2 - min, etc.) A quick way to calculate the log2 floor of a number is to count the
 * leading zeros in its binary representation: for a 32/64 bit number, the log2 floor is "31/63 -
 * the number of leading zeros". `countl_zero` does exactly that (clz = count leading zeros). Notice
 * that `countl_zero` is undefined for 0 (as well as log2 of 0). Our first index is 1, so we don't
 * need to worry about that. since we only care about the parity of the log2 floor, we can just
 * check the LSB of the number of leading zeros (N is the number of bits in the number, which is
 * even):
 *
 * n is a min node                      <=>
 * log2(n) % 2 == 0                     <=>
 * (N - 1 - countl_zero(n)) % 2 == 0    <=>
 * countl_zero(n) % 2 == 1
 *
 * So we can simply check for `(countl_zero(n) & 1)`.
 *
 * Additional info:
 *    Correctness: https://godbolt.org/z/W7n9e39qj
 *    Optimality:  https://quick-bench.com/q/Rl3sUfldpGlhQWjXopnTtxh95kI
 */
template <typename T, typename Compare>
bool min_max_heap<T, Compare>::is_min(size_t index) const {
    return std::countl_zero(index) & 1;
}

template <typename T, typename Compare>
size_t min_max_heap<T, Compare>::parent(size_t index) const {
    return index >> 1;
}

template <typename T, typename Compare>
size_t min_max_heap<T, Compare>::left_child(size_t index) const {
    return index << 1;
}

template <typename T, typename Compare>
size_t min_max_heap<T, Compare>::right_child(size_t index) const {
    return (index << 1) + 1;
}

template <typename T, typename Compare>
void min_max_heap<T, Compare>::swap(size_t index1, size_t index2) {
    std::swap(data[index1], data[index2]);
}

template <typename T, typename Compare>
void min_max_heap<T, Compare>::bubble_up(size_t idx) {
    size_t p_idx = parent(idx);
    if (!p_idx)
        return;

    if (is_min(idx)) {
        if (compare(p_idx, idx)) {
            swap(idx, p_idx);
            bubble_up_kind<false>(p_idx); // bubble up max
        } else {
            bubble_up_kind<true>(idx); // bubble up min
        }
    } else {
        if (compare(idx, p_idx)) {
            swap(idx, p_idx);
            bubble_up_kind<true>(p_idx); // bubble up min
        } else {
            bubble_up_kind<false>(idx); // bubble up max
        }
    }
}

template <typename T, typename Compare>
template <bool min>
void min_max_heap<T, Compare>::bubble_up_kind(size_t idx) {
    size_t gp_idx = parent(parent(idx));
    if (!gp_idx)
        return;

    if (min ? compare(idx, gp_idx) : compare(gp_idx, idx)) {
        swap(idx, gp_idx);
        bubble_up_kind<min>(gp_idx);
    }
}

template <typename T, typename Compare>
char min_max_heap<T, Compare>::highest_descendant_in_range(size_t idx) const {
    size_t a = left_child(idx);
    size_t b = right_child(idx);
    size_t c = left_child(a);
    size_t d = right_child(a);
    size_t e = left_child(b);
    size_t f = right_child(b);

    if (f < data.size())
        return 0xf;
    if (e < data.size())
        return 0xe;
    if (d < data.size())
        return 0xd;
    if (c < data.size())
        return 0xc;
    if (b < data.size())
        return 0xb;
    if (a < data.size())
        return 0xa;

    return 0x0;
}

// basing on the min/max heap property, we can determine the best child/grandchild out of the
// existing ones without having to Compare all of them
template <typename T, typename Compare>
template <bool min>
size_t min_max_heap<T, Compare>::index_best_child_grandchild(size_t idx) const {
    size_t a = left_child(idx);
    size_t b = right_child(idx);
    size_t c = left_child(a);
    size_t d = right_child(a);
    size_t e = left_child(b);
    size_t f = right_child(b);

    switch (highest_descendant_in_range(idx)) {
    case 0xf:
        return choose_from_4<min>(c, d, e, f);
    case 0xe:
        return choose_from_3<min>(c, d, e);
    case 0xd:
        return choose_from_3<min>(b, c, d);
    case 0xc:
        return choose_from_2<min>(b, c);
    case 0xb:
        return choose_from_2<min>(a, b);
    case 0xa:
        return a;
    default:
        return -1;
    }
}

template <typename T, typename Compare>
template <bool min>
void min_max_heap<T, Compare>::trickle_down(size_t idx) {
    size_t best = index_best_child_grandchild<min>(idx);
    if (best == -1)
        return;
    auto cmp = compare;

    if (best > right_child(idx)) {
        // best is a grandchild
        if (min ? cmp(idx, best) : cmp(best, idx)) {
            swap(idx, best);
            if (min ? cmp(parent(best), best) : cmp(best, parent(best))) {
                swap(best, parent(best));
            }
            trickle_down<min>(best);
        }
    } else {
        // best is a child
        if (min ? cmp(idx, best) : cmp(best, idx))
            swap(idx, best);
    }
}

/***************************** Public Implementation *****************************/

template <typename T, typename Compare>
void min_max_heap<T, Compare>::insert(const T &value) {
    data.push_back(value);
    bubble_up(size());
}

template <typename T, typename Compare>
T min_max_heap<T, Compare>::pop_min() {
    assert(size());

    T min = data[1];
    data[1] = data.back();
    data.pop_back();

    if (size())
        trickle_down<true>(1);

    return min;
}

template <typename T, typename Compare>
T min_max_heap<T, Compare>::pop_max() {
    assert(size());

    if (size() < 3) {
        T max = data[size()];
        data.pop_back();
        return max;
    }

    size_t max_idx = compare(data[3], data[2]) ? 2 : 3;

    T max = data[max_idx];
    data[max_idx] = data.back();
    data.pop_back();

    trickle_down<false>(max_idx);

    return max;
}

template <typename T, typename Compare>
const T &min_max_heap<T, Compare>::peek_min() const {
    assert(size());
    return data[1];
}

template <typename T, typename Compare>
const T &min_max_heap<T, Compare>::peek_max() const {
    assert(size());
    return (size() < 3 || compare(3, 2)) ? data[2] : data[3];
}

template <typename T, typename Compare>
T min_max_heap<T, Compare>::exchange_min(const T &value) {
    assert(size());
    T min = data[1];
    data[1] = value;
    trickle_down<true>(1);
    return min;
}

template <typename T, typename Compare>
T min_max_heap<T, Compare>::exchange_max(const T &value) {
    assert(size());

    switch (size()) {
    case 1: {
        T max = data[1];
        data[1] = value;
        return max;
    }

    case 2: {
        T max = data[2];
        data[2] = value;
        // if the new value is smaller than the parent (root), perform a single-step bubble up
        if (compare(data[2], data[1]))
            std::swap(data[1], data[2]);
        return max;
    }

    default: {
        size_t max_idx = compare(data[3], data[2]) ? 2 : 3;
        T max = data[max_idx];
        data[max_idx] = value;
        // if the new value is smaller than the parent (root), perform a single-step bubble up
        if (compare(data[max_idx], data[1]))
            std::swap(data[1], data[max_idx]);
        trickle_down<false>(max_idx);
        return max;
    }

    }
}

} // namespace vecsim_stl
