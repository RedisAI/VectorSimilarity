#include "visited_nodes_handler.h"

namespace hnswlib {

VisitedNodesHandler::VisitedNodesHandler(unsigned int cap,
                                         const std::shared_ptr<VecSimAllocator> &allocator)
    : VecsimBaseObject(allocator) {
    cur_tag = 0;
    num_elements = cap;
    elements_tags = reinterpret_cast<tag_t *>(allocator->callocate(sizeof(tag_t) * num_elements));
}

tag_t VisitedNodesHandler::getFreshTag() {
    cur_tag++;
    if (cur_tag == 0) {
        memset(elements_tags, 0, sizeof(tag_t) * num_elements);
        cur_tag++;
    }
    return cur_tag;
}

VisitedNodesHandler::~VisitedNodesHandler() { allocator->free_allocation(elements_tags); }

/**
 * VisitedNodesHandlerPool Methods (when parallelization is enabled)
 */
VisitedNodesHandlerPool::VisitedNodesHandlerPool(int initial_pool_size, int cap,
                                                 const std::shared_ptr<VecSimAllocator> &allocator)
    : VecsimBaseObject(allocator), pool(allocator), num_elements(cap) {
    for (int i = 0; i < initial_pool_size; i++)
        pool.push_front(new (allocator) VisitedNodesHandler(cap, allocator));
}

VisitedNodesHandler *VisitedNodesHandlerPool::getAvailableVisitedNodesHandler() {
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

void VisitedNodesHandlerPool::returnVisitedNodesHandlerToPool(VisitedNodesHandler *handler) {
    std::unique_lock<std::mutex> lock(pool_guard);
    pool.push_front(handler);
}

VisitedNodesHandlerPool::~VisitedNodesHandlerPool() {
    while (!pool.empty()) {
        VisitedNodesHandler *handler = pool.front();
        pool.pop_front();
        delete handler;
    }
}

} // namespace hnswlib
