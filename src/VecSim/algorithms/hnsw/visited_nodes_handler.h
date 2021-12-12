#pragma once

#include <mutex>
#include <deque>

#include "VecSim/memory/vecsim_malloc.h"

namespace hnswlib {

typedef ushort tag_t;

/**
 * Used as a singleton that is responsible for marking nodes that were visited in the graph scan.
 * Every scan has a "pseudo unique" tag which is associated with this specific scan, and nodes
 * that were visited in this particular scan are tagged with this tag. The tags range from
 * 1-MAX_USHORT, and we reset the tags after we complete MAX_USHORT scans.
 */
class VisitedNodesHandler : public VecsimBaseObject {
    tag_t cur_tag;
    tag_t *elements_tags;
    uint num_elements;

public:
    VisitedNodesHandler(uint cap, const std::shared_ptr<VecSimAllocator> &allocator)
        : VecsimBaseObject(allocator) {
        cur_tag = 0;
        num_elements = cap;
        elements_tags =
            reinterpret_cast<tag_t *>(allocator->allocate(sizeof(tag_t) * num_elements));
        memset(elements_tags, 0, sizeof(tag_t) * num_elements);
    }

    tag_t getFreshTag() {
        cur_tag++;
        if (cur_tag == 0) {
            memset(elements_tags, 0, sizeof(tag_t) * num_elements);
            cur_tag++;
        }
        return cur_tag;
    }

    void visitNode(uint node_id, tag_t tag) { elements_tags[node_id] = tag; }

    tag_t getNodeTag(uint node_id) { return elements_tags[node_id]; }

    ~VisitedNodesHandler() override { allocator->free_allocation(elements_tags); }
};

/**
 * A wrapper class for using a pool of VisitedNodesHandler (relevant only when parallelization is
 * enabled).
 */
class VisitedNodesHandlerPool : public VecsimBaseObject {
    std::deque<VisitedNodesHandler *, VecsimSTLAllocator<VisitedNodesHandler *>> pool;
    std::mutex pool_guard;
    uint num_elements;

public:
    VisitedNodesHandlerPool(int initial_pool_size, int cap,
                            const std::shared_ptr<VecSimAllocator> &allocator)
        : VecsimBaseObject(allocator), pool(allocator), num_elements(cap) {
        for (int i = 0; i < initial_pool_size; i++)
            pool.push_front(new (allocator) VisitedNodesHandler(cap, allocator));
    }

    VisitedNodesHandler *getAvailableVisitedNodesHandler() {
        VisitedNodesHandler *handler;
        std::unique_lock<std::mutex> lock(pool_guard);
        if (!pool.empty()) {
            handler = pool.front();
            pool.pop_front();
        } else {
            handler = new (allocator) VisitedNodesHandler(this->num_elements, this->allocator);
        }
        return handler;
    }

    void returnVisitedNodesHandlerToPool(VisitedNodesHandler *handler) {
        std::unique_lock<std::mutex> lock(pool_guard);
        pool.push_front(handler);
    }

    ~VisitedNodesHandlerPool() override {
        while (!pool.empty()) {
            VisitedNodesHandler *handler = pool.front();
            pool.pop_front();
            delete handler;
        }
    }
};

} // namespace hnswlib
