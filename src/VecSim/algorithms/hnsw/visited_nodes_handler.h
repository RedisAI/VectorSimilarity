#pragma once

#include <mutex>
#include <deque>
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/memory/vecsim_base.h"

namespace hnswlib {

typedef ushort tag_t;

/**
 * Used as a singleton that is responsible for marking nodes that were visited in the graph scan.
 * Every scan has a "pseudo unique" tag which is associated with this specific scan, and nodes
 * that were visited in this particular scan are tagged with this tag. The tags range from
 * 1-MAX_USHORT, and we reset the tags after we complete MAX_USHORT scans.
 */
class VisitedNodesHandler : public VecsimBaseObject {
private:
    tag_t cur_tag;
    tag_t *elements_tags;
    uint num_elements;

public:
    VisitedNodesHandler(uint cap, const std::shared_ptr<VecSimAllocator> &allocator);

    // Return unused tag for marking the visited nodes. The tags are cyclic, so whenever we reach
    // zero, we reset the tags of all the nodes (and use 1 as the fresh tag)
    tag_t getFreshTag();

    inline tag_t *getElementsTags() { return elements_tags; }

    void reset();

    // Mark node_id with tag, to have an indication that this node has been visited.
    inline void tagNode(uint node_id, tag_t tag) { elements_tags[node_id] = tag; }

    // Get the tag in which node_id is marked currently.
    inline tag_t getNodeTag(uint node_id) { return elements_tags[node_id]; }

    ~VisitedNodesHandler() override;
};

/**
 * A wrapper class for using a pool of VisitedNodesHandler (relevant only when parallelization is
 * enabled).
 */
class VisitedNodesHandlerPool : public VecsimBaseObject {
private:
    std::deque<VisitedNodesHandler *, VecsimSTLAllocator<VisitedNodesHandler *>> pool;
    std::mutex pool_guard;
    uint num_elements;

public:
    VisitedNodesHandlerPool(int initial_pool_size, int cap,
                            const std::shared_ptr<VecSimAllocator> &allocator);

    VisitedNodesHandler *getAvailableVisitedNodesHandler();

    void returnVisitedNodesHandlerToPool(VisitedNodesHandler *handler);

    ~VisitedNodesHandlerPool() override;
};

} // namespace hnswlib
