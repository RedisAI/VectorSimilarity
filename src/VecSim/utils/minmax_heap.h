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
class min_max_heap : public abstract_min_max_heap<T> {
private:
    vecsim_stl::vector<T> data;

private:
    /*
     * `is_min` returns true if the index is a min node, false otherwise.
     * A node is a min node if its level (depth) is odd (and the root has a depth 1).
     * With our array representation, a node is a min node if the log2 floor of its index is even.
     * (log2 floor of 1 is 0 - min, log2 floor of 2 is 1 - max, log2 floor of 3 is 1 - max, log2
     * floor of 4 is 2 - min, etc.) A quick way to calculate the log2 floor of a number is to count
     * the leading zeros in its binary representation: for a 32/64 bit number, the log2 floor is
     * "31/63 - the number of leading zeros". `countl_zero` does exactly that. since we only care
     * about the parity of the log2 floor, we can just check the LSB of the number of leading zeros
     * (N is the number of bits in the number, which is even):
     *
     * n is a min node                      if and only if
     * log2(n) % 2 == 0                     if and only if
     * (N - 1 - countl_zero(n)) % 2 == 0    if and only if
     * countl_zero(n) % 2 == 1
     *
     * So we can simply check for `(countl_zero(n) & 1)`.
     *
     * Additional info:
     *    Correctness: https://godbolt.org/z/W7n9e39qj
     *    Optimality:  https://quick-bench.com/q/Rl3sUfldpGlhQWjXopnTtxh95kI
     */
    inline bool is_min(size_t index) const { return std::countl_zero(index) & 1; }

    inline size_t parent(size_t index) const { return index >> 1; }

    inline size_t left_child(size_t index) const { return index << 1; }

    inline size_t right_child(size_t index) const { return (index << 1) + 1; }

    inline void swap(size_t index1, size_t index2) { std::swap(data[index1], data[index2]); }

    inline bool compare(size_t index1, size_t index2) const {
        return Compare()(data[index1], data[index2]);
    }

    inline void bubble_up_new() {
        size_t idx = size();
        size_t p_idx = parent(idx);
        if (!p_idx)
            return;

        if (is_min(idx)) {
            if (compare(p_idx, idx)) {
                swap(idx, p_idx);
                bubble_up<false>(p_idx); // bubble up max
            } else {
                bubble_up<true>(idx); // bubble up min
            }
        } else {
            if (compare(idx, p_idx)) {
                swap(idx, p_idx);
                bubble_up<true>(p_idx); // bubble up min
            } else {
                bubble_up<false>(idx); // bubble up max
            }
        }
    }

    template <bool min>
    void bubble_up(size_t idx) {
        size_t gp_idx = parent(parent(idx));
        if (!gp_idx)
            return;

        if (min ? compare(idx, gp_idx) : compare(gp_idx, idx)) {
            swap(idx, gp_idx);
            bubble_up<min>(gp_idx);
        }
    }

    inline char highest_descendant_in_range(size_t idx) const {
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
    template <bool min>
    inline size_t index_best_child_grandchild(size_t idx) const {
        size_t a = left_child(idx);
        size_t b = right_child(idx);
        size_t c = left_child(a);
        size_t d = right_child(a);
        size_t e = left_child(b);
        size_t f = right_child(b);

        switch (highest_descendant_in_range(idx)) {
        case 0xf:
            return choose_from<min>(c, d, e, f);
        case 0xe:
            return choose_from<min>(c, d, e);
        case 0xd:
            return choose_from<min>(b, c, d);
        case 0xc:
            return choose_from<min>(b, c);
        case 0xb:
            return choose_from<min>(a, b);
        case 0xa:
            return a;
        default:
            return SIZE_MAX;
        }
    }

    template <bool min>
    void trickle_down(size_t idx) {
        size_t best = index_best_child_grandchild<min>(idx);
        if (best == SIZE_MAX)
            return;

        if (best > right_child(idx)) {
            // best is a grandchild
            if (min ? compare(best, idx) : compare(idx, best)) {
                swap(idx, best);
                if (min ? compare(parent(best), best) : compare(best, parent(best))) {
                    swap(best, parent(best));
                }
                trickle_down<min>(best);
            }
        } else {
            // best is a child
            if (min ? compare(best, idx) : compare(idx, best))
                swap(idx, best);
        }
    }

    // choose best index from a list of indices of any size.
    // min = true:  choose the index of the minimal value.
    // min = false: choose the index of the maximal value.
    template <bool min>
    inline size_t choose_from(size_t i) const {
        return i;
    }
    template <bool min, typename... Args>
    inline size_t choose_from(size_t i, Args... args) const {
        size_t j = choose_from<min>(args...);
        return (min ? compare(i, j) : compare(j, i)) ? i : j;
    }

    /***************************** Public Implementation *****************************/
public:
    min_max_heap(const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_min_max_heap<T>(alloc), data(alloc) {
        data.push_back(T()); // dummy element
    }
    min_max_heap(size_t cap, const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_min_max_heap<T>(alloc), data(alloc) {
        data.reserve(cap + 1);
        data.push_back(T()); // dummy element
    }
    ~min_max_heap() = default;

    inline size_t size() const override { return data.size() - 1; }
    inline bool empty() const override { return size() == 0; }
    inline void clear() {
        data.clear();
        data.push_back(T()); // dummy element
    }

    inline void insert(const T &value) {
        data.push_back(value);
        bubble_up_new();
    }

    inline T pop_min() {
        assert(size());

        T min = data[1];
        data[1] = data.back();
        data.pop_back();

        trickle_down<true>(1);

        return min;
    }

    inline T pop_max() {
        assert(size());

        if (size() < 3) {
            T max = data[size()];
            data.pop_back();
            return max;
        }

        size_t max_idx = choose_from<false>(2, 3);

        T max = data[max_idx];
        data[max_idx] = data.back();
        data.pop_back();

        trickle_down<false>(max_idx);

        return max;
    }

    inline const T &peek_min() const {
        assert(size());
        return data[1];
    }

    inline const T &peek_max() const {
        assert(size());
        if (size() < 3)
            return data[size()];
        return compare(3, 2) ? data[2] : data[3];
    }

    // combines pop-and-then-insert logic
    inline T exchange_min(const T &value) {
        assert(size());
        T min = data[1];
        data[1] = value;
        trickle_down<true>(1);
        return min;
    }

    // combines pop-and-then-insert logic
    inline T exchange_max(const T &value) {
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
            if (compare(2, 1))
                swap(1, 2);
            return max;
        }

        default: {
            size_t max_idx = choose_from<false>(2, 3);
            T max = data[max_idx];
            data[max_idx] = value;
            // if the new value is smaller than the parent (root), perform a single-step bubble up
            if (compare(max_idx, 1))
                swap(1, max_idx);
            trickle_down<false>(max_idx);
            return max;
        }
        }
    }

    // Convenience functions for emplacing elements

    template <typename... Args>
    inline void emplace(Args &&...args) {
        data.emplace_back(std::forward<Args>(args)...);
        bubble_up_new();
    }

    template <typename... Args>
    inline T exchange_min(Args &&...args) {
        return exchange_min(static_cast<const T &>(T(args...)));
    }

    template <typename... Args>
    inline T exchange_max(Args &&...args) {
        return exchange_max(static_cast<const T &>(T(args...)));
    }
}; // min_max_heap

} // namespace vecsim_stl
