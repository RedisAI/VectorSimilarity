/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "VecSim/utils/arr_cpp.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/query_result_struct.h"

namespace vecsim_stl {

// An abstract API for query result container, used by RANGE queries.
struct abstract_results_container : public VecsimBaseObject {
public:
    abstract_results_container(const std::shared_ptr<VecSimAllocator> &alloc)
        : VecsimBaseObject(alloc) {}
    ~abstract_results_container() = default;

    // Inserts (or updates) a new result to the container.
    virtual inline void emplace(size_t id, double score) = 0;

    // Returns the size of the container
    virtual inline size_t size() const = 0;

    // Returns an array (arr_cpp.h) containing all current data, and passes its ownership
    virtual inline VecSimQueryResult *get_results() = 0;
};

struct unique_results_container : public abstract_results_container {
private:
    vecsim_stl::unordered_map<size_t, double> idToScore;

public:
    explicit unique_results_container(const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_results_container(alloc), idToScore(alloc) {}
    explicit unique_results_container(size_t cap, const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_results_container(alloc), idToScore(cap, alloc) {}

    inline void emplace(size_t id, double score) override {
        auto existing = idToScore.find(id);
        if (existing == idToScore.end()) {
            idToScore.emplace(id, score);
        } else if (existing->second > score) {
            existing->second = score;
        }
    }

    inline size_t size() const override { return idToScore.size(); }

    inline VecSimQueryResult *get_results() override {
        auto *data = array_new_len<VecSimQueryResult>(idToScore.size(), idToScore.size());
        size_t index = 0;
        for (auto res : idToScore) {
            VecSimQueryResult_SetId(data[index], res.first);
            VecSimQueryResult_SetScore(data[index], res.second);
            index++;
        }
        return data;
    }
};

struct default_results_container : public abstract_results_container {
private:
    VecSimQueryResult *_data;
    bool owns_data;

public:
    explicit default_results_container(const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_results_container(alloc), _data(array_new<VecSimQueryResult>(0)),
          owns_data(true) {}
    explicit default_results_container(size_t cap, const std::shared_ptr<VecSimAllocator> &alloc)
        : abstract_results_container(alloc), _data(array_new<VecSimQueryResult>(cap)),
          owns_data(true) {}

    ~default_results_container() {
        if (owns_data)
            array_free(_data);
    }

    inline void emplace(size_t id, double score) override {
        auto res = VecSimQueryResult{id, score};
        _data = array_append(_data, res);
    }
    inline size_t size() const override { return array_len(_data); }
    inline VecSimQueryResult *get_results() override {
        owns_data = false;
        return _data;
    }
};
} // namespace vecsim_stl
