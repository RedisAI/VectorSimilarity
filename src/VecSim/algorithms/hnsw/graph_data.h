
#pragma once

#include <cassert>
#include <algorithm>
#include <span>
#include <stdexcept>
#include "VecSim/utils/vec_utils.h"

// Amortized shrink thresholds for incoming edges vectors.
// Shrink is triggered when: capacity > SHRINK_RATIO * size
constexpr size_t INCOMING_EDGES_SHRINK_RATIO = 2;

template <typename DistType>
using candidatesList = vecsim_stl::vector<std::pair<DistType, idType>>;

typedef uint16_t linkListSize;

// The unidirectional incoming edges of one element at one level, stored as a single allocation of
// this header followed by the ids.
struct IncomingEdges {
    uint32_t size;
    uint32_t capacity;
    idType ids[];
};

struct ElementLevelData {
    // The ids of the elements that point to this one where the edge is *unidirectional*, or null
    // while there are none. Owned by this level data and freed with `freeIncomingEdges`. It is
    // guarded by the element's lock, like the links.
    IncomingEdges *incomingUnidirectionalEdges;
    linkListSize numLinks;
    // Flexible array member - https://en.wikipedia.org/wiki/Flexible_array_member
    // Using this trick, we can have the links list as part of the ElementLevelData struct, and
    // avoid the need to dereference a pointer to get to the links list. We have to calculate the
    // size of the struct manually, as `sizeof(ElementLevelData)` will not include this member. We
    // do so in the constructor of the index, under the name `levelDataSize` (and
    // `elementGraphDataSize`). Notice that this member must be the last member of the struct and
    // all nesting structs.
    idType links[];

    ElementLevelData() : incomingUnidirectionalEdges(nullptr), numLinks(0) {}

    linkListSize getNumLinks() const { return this->numLinks; }
    idType getLinkAtPos(size_t pos) const {
        assert(pos < numLinks);
        return this->links[pos];
    }
    std::span<const idType> getIncomingEdges() const {
        if (!incomingUnidirectionalEdges) {
            return {};
        }
        return {incomingUnidirectionalEdges->ids, incomingUnidirectionalEdges->size};
    }
    size_t getIncomingEdgesCapacity() const {
        return incomingUnidirectionalEdges ? incomingUnidirectionalEdges->capacity : 0;
    }
    std::vector<idType> copyLinks() {
        std::vector<idType> links_copy;
        links_copy.assign(links, links + numLinks);
        return links_copy;
    }
    // Sets the outgoing links of the current element.
    // Assumes that the object has the capacity to hold all the links.
    void setLinks(vecsim_stl::vector<idType> &links) {
        numLinks = links.size();
        memcpy(this->links, links.data(), numLinks * sizeof(idType));
    }
    template <typename DistType>
    void setLinks(candidatesList<DistType> &links) {
        numLinks = 0;
        for (auto &link : links) {
            this->links[numLinks++] = link.second;
        }
    }
    void popLink() { this->numLinks--; }
    void setNumLinks(linkListSize num) { this->numLinks = num; }
    void setLinkAtPos(size_t pos, idType node_id) { this->links[pos] = node_id; }
    void appendLink(idType node_id) { this->links[this->numLinks++] = node_id; }
    void removeLink(idType node_id) {
        size_t i = 0;
        for (; i < numLinks; i++) {
            if (links[i] == node_id) {
                links[i] = links[numLinks - 1];
                break;
            }
        }
        assert(i < numLinks && "Corruption in HNSW index"); // node_id not found - error
        numLinks--;
    }
    // Grows the capacity like `std::vector` does (doubling, starting at one).
    void newIncomingUnidirectionalEdge(idType node_id, VecSimAllocator &allocator) {
        size_t size = getIncomingEdges().size();
        if (size == getIncomingEdgesCapacity()) {
            resizeIncomingEdges(std::max<size_t>(1, 2 * size), allocator);
        }
        incomingUnidirectionalEdges->ids[incomingUnidirectionalEdges->size++] = node_id;
    }
    // Removes node_id by moving the last id into its place. Shrinks the allocation when it is more
    // than INCOMING_EDGES_SHRINK_RATIO times larger than needed, and frees it when it empties.
    bool removeIncomingUnidirectionalEdgeIfExists(idType node_id, VecSimAllocator &allocator) {
        auto edges = getIncomingEdges();
        auto it = std::find(edges.begin(), edges.end(), node_id);
        if (it == edges.end()) {
            return false;
        }
        incomingUnidirectionalEdges->ids[it - edges.begin()] = edges.back();
        size_t size = --incomingUnidirectionalEdges->size;
        if (getIncomingEdgesCapacity() > INCOMING_EDGES_SHRINK_RATIO * size) {
            resizeIncomingEdges(size, allocator);
        }
        return true;
    }
    void swapNodeIdInIncomingEdges(idType id_before, idType id_after) {
        auto edges = getIncomingEdges();
        auto it = std::find(edges.begin(), edges.end(), id_before);
        // This should always succeed
        assert(it != edges.end());
        incomingUnidirectionalEdges->ids[it - edges.begin()] = id_after;
    }
    // Makes room for at least `capacity` incoming edges in one allocation.
    void reserveIncomingEdges(size_t capacity, VecSimAllocator &allocator) {
        if (capacity > getIncomingEdgesCapacity()) {
            resizeIncomingEdges(capacity, allocator);
        }
    }
    void shrinkIncomingEdgesToFit(VecSimAllocator &allocator) {
        if (getIncomingEdgesCapacity() > getIncomingEdges().size()) {
            resizeIncomingEdges(getIncomingEdges().size(), allocator);
        }
    }
    void freeIncomingEdges(VecSimAllocator &allocator) {
        allocator.free_allocation(incomingUnidirectionalEdges);
        incomingUnidirectionalEdges = nullptr;
    }

private:
    // Sets the capacity to `capacity` (at least the current size), keeping the ids.
    void resizeIncomingEdges(size_t capacity, VecSimAllocator &allocator) {
        assert(capacity >= getIncomingEdges().size());
        if (capacity == 0) {
            freeIncomingEdges(allocator);
            return;
        }
        bool was_empty = incomingUnidirectionalEdges == nullptr;
        auto *resized = static_cast<IncomingEdges *>(allocator.reallocate(
            incomingUnidirectionalEdges, sizeof(IncomingEdges) + capacity * sizeof(idType)));
        if (resized == nullptr) {
            throw std::runtime_error("VecSim index low memory error");
        }
        if (was_empty) {
            resized->size = 0;
        }
        resized->capacity = capacity;
        incomingUnidirectionalEdges = resized;
    }
};

struct ElementGraphData {
    size_t toplevel;
    ElementLevelData *others;
    ElementLevelData level0;

    ElementGraphData(size_t maxLevel, size_t high_level_size,
                     std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0() {
        if (toplevel > 0) {
            others = (ElementLevelData *)allocator->callocate(high_level_size * toplevel);
            if (others == nullptr) {
                throw std::runtime_error("VecSim index low memory error");
            }
            for (size_t i = 0; i < maxLevel; i++) {
                new ((char *)others + i * high_level_size) ElementLevelData();
            }
        }
    }
    ~ElementGraphData() = delete; // should be destroyed using `destroy'

    void destroy(size_t levelDataSize, std::shared_ptr<VecSimAllocator> allocator) {
        this->level0.freeIncomingEdges(*allocator);
        ElementLevelData *cur_ld = this->others;
        for (size_t i = 0; i < this->toplevel; i++) {
            cur_ld->freeIncomingEdges(*allocator);
            cur_ld = reinterpret_cast<ElementLevelData *>(reinterpret_cast<char *>(cur_ld) +
                                                          levelDataSize);
        }
        allocator->free_allocation(this->others);
    }
    ElementLevelData &getElementLevelData(size_t level, size_t levelDataSize) {
        assert(level <= this->toplevel);
        if (level == 0) {
            return this->level0;
        }
        return *reinterpret_cast<ElementLevelData *>(reinterpret_cast<char *>(this->others) +
                                                     (level - 1) * levelDataSize);
    }
};
