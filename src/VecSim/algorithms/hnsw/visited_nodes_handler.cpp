/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include <cassert>
#include "visited_nodes_handler.h"

VisitedNodesHandler::VisitedNodesHandler(unsigned int cap,
                                         const std::shared_ptr<VecSimAllocator> &allocator)
    : VecsimBaseObject(allocator) {
    cur_tag = 0;
    num_elements = cap;
    elements_tags = reinterpret_cast<tag_t *>(allocator->callocate(sizeof(tag_t) * num_elements));
}

void VisitedNodesHandler::reset() {
    memset(elements_tags, 0, sizeof(tag_t) * num_elements);
    cur_tag = 0;
}

void VisitedNodesHandler::resize(size_t new_size) {
    this->num_elements = new_size;
    this->elements_tags = reinterpret_cast<tag_t *>(
        allocator->reallocate(this->elements_tags, sizeof(tag_t) * new_size));
    this->reset();
}

tag_t VisitedNodesHandler::getFreshTag() {
    cur_tag++;
    if (cur_tag == 0) {
        this->reset();
        cur_tag++;
    }
    return cur_tag;
}

VisitedNodesHandler::~VisitedNodesHandler() { allocator->free_allocation(elements_tags); }

/**
 * VisitedNodesHandlerPool methods to enable parallel graph scans.
 */
VisitedNodesHandlerPool::VisitedNodesHandlerPool(size_t initial_pool_size, int cap,
                                                 const std::shared_ptr<VecSimAllocator> &allocator)
    : VecsimBaseObject(allocator), pool(initial_pool_size, allocator), num_elements(cap),
      total_handlers_in_use(1) {
    for (size_t i = 0; i < initial_pool_size; i++)
        pool[i] = new (allocator) VisitedNodesHandler(cap, allocator);
}

VisitedNodesHandler *VisitedNodesHandlerPool::getAvailableVisitedNodesHandler() {
    VisitedNodesHandler *handler;
    std::unique_lock<std::mutex> lock(pool_guard);
    if (!pool.empty()) {
        handler = pool.back();
        pool.pop_back();
    } else {
        handler = new (allocator) VisitedNodesHandler(this->num_elements, this->allocator);
        total_handlers_in_use++;
    }
    return handler;
}

void VisitedNodesHandlerPool::returnVisitedNodesHandlerToPool(VisitedNodesHandler *handler) {
    std::unique_lock<std::mutex> lock(pool_guard);
    pool.push_back(handler);
    pool.shrink_to_fit();
}

void VisitedNodesHandlerPool::resize(size_t new_size) {
    assert(total_handlers_in_use ==
           pool.size()); // validate that there is no handlers in use outside the pool.
    this->num_elements = new_size;
    for (auto &handler : this->pool) {
        handler->resize(new_size);
    }
}

VisitedNodesHandlerPool::~VisitedNodesHandlerPool() {
    while (!pool.empty()) {
        VisitedNodesHandler *handler = pool.back();
        pool.pop_back();
        delete handler;
    }
}
