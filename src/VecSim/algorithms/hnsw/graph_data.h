
#pragma once

#include <cassert>
#include <algorithm>
#include <mutex>
#include "VecSim/utils/vec_utils.h"

// Amortized shrink thresholds for incoming edges vectors.
// Shrink is triggered when: capacity > SHRINK_RATIO * size
constexpr size_t INCOMING_EDGES_SHRINK_RATIO = 2;

template <typename DistType>
using candidatesList = vecsim_stl::vector<std::pair<DistType, idType>>;

typedef uint16_t linkListSize;

struct ElementLevelData {
    // A list of ids that are pointing to the node where each edge is *unidirectional*
    vecsim_stl::vector<idType> *incomingUnidirectionalEdges;
    linkListSize numLinks;
    // Flexible array member - https://en.wikipedia.org/wiki/Flexible_array_member
    // Using this trick, we can have the links list as part of the ElementLevelData struct, and
    // avoid the need to dereference a pointer to get to the links list. We have to calculate the
    // size of the struct manually, as `sizeof(ElementLevelData)` will not include this member. We
    // do so in the constructor of the index, under the name `levelDataSize` (and
    // `elementGraphDataSize`). Notice that this member must be the last member of the struct and
    // all nesting structs.
    idType links[];

    explicit ElementLevelData(std::shared_ptr<VecSimAllocator> allocator)
        : incomingUnidirectionalEdges(new(allocator) vecsim_stl::vector<idType>(allocator)),
          numLinks(0) {}

    linkListSize getNumLinks() const { return this->numLinks; }
    idType getLinkAtPos(size_t pos) const {
        assert(pos < numLinks);
        return this->links[pos];
    }
    const vecsim_stl::vector<idType> &getIncomingEdges() const {
        return *incomingUnidirectionalEdges;
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
    void newIncomingUnidirectionalEdge(idType node_id) {
        this->incomingUnidirectionalEdges->push_back(node_id);
    }
    bool removeIncomingUnidirectionalEdgeIfExists(idType node_id) {
        bool result = this->incomingUnidirectionalEdges->remove(node_id);

        if (result) {
            auto &vec = *this->incomingUnidirectionalEdges;
            if (vec.capacity() > INCOMING_EDGES_SHRINK_RATIO * vec.size()) {
                vec.shrink_to_fit();
            }
        }

        return result;
    }
    void swapNodeIdInIncomingEdges(idType id_before, idType id_after) {
        auto it = std::find(this->incomingUnidirectionalEdges->begin(),
                            this->incomingUnidirectionalEdges->end(), id_before);
        // This should always succeed
        assert(it != this->incomingUnidirectionalEdges->end());
        *it = id_after;
    }
    // Deep-copy this level record into `dst`. `recordSize` is the full byte size
    // of the record including the `links` flexible-array member (`levelDataSize`
    // for upper levels; the level-0 region size for level 0). The destination
    // gets an INDEPENDENT incoming-edges vector so a snapshot never shares a
    // mutable vector with the live index. Used by block-level copy-on-write.
    void copyInto(ElementLevelData *dst, size_t recordSize,
                  const std::shared_ptr<VecSimAllocator> &allocator) const {
        // Copies numLinks + the links FAM in one shot (also shallow-copies the
        // incoming-edges pointer, which we immediately replace below).
        memcpy(dst, this, recordSize);
        dst->incomingUnidirectionalEdges =
            new (allocator) vecsim_stl::vector<idType>(*this->incomingUnidirectionalEdges);
    }
};

struct ElementGraphData {
    size_t toplevel;
    std::mutex neighborsGuard;
    ElementLevelData *others;
    ElementLevelData level0;

    ElementGraphData(size_t maxLevel, size_t high_level_size,
                     std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0(allocator) {
        if (toplevel > 0) {
            others = (ElementLevelData *)allocator->callocate(high_level_size * toplevel);
            if (others == nullptr) {
                throw std::runtime_error("VecSim index low memory error");
            }
            for (size_t i = 0; i < maxLevel; i++) {
                new ((char *)others + i * high_level_size) ElementLevelData(allocator);
            }
        }
    }
    ~ElementGraphData() = delete; // should be destroyed using `destroy'

    void destroy(size_t levelDataSize, std::shared_ptr<VecSimAllocator> allocator) {
        delete this->level0.incomingUnidirectionalEdges;
        ElementLevelData *cur_ld = this->others;
        for (size_t i = 0; i < this->toplevel; i++) {
            delete cur_ld->incomingUnidirectionalEdges;
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

    // Deep-copy this element's graph data into raw, zeroed memory `dst` of size
    // `elementGraphDataSize`. The copy gets a freshly-constructed mutex (lock
    // state is never copied) and independent per-level incoming-edges vectors,
    // so it shares no mutable state with the source. This is the foundational
    // operation for block-level copy-on-write: it materializes an immutable
    // snapshot version of a node that the live index can keep mutating.
    void copyTo(ElementGraphData *dst, size_t levelDataSize, size_t elementGraphDataSize,
                const std::shared_ptr<VecSimAllocator> &allocator) const {
        dst->toplevel = this->toplevel;
        // Fresh mutex; a snapshot reads immutable data and never locks, and the
        // source's lock state must never be aliased into the copy.
        new (&dst->neighborsGuard) std::mutex();
        // Level 0 is inline; its links FAM occupies the tail of the element, so
        // its record size is everything past the offset of `level0`.
        const size_t level0Size =
            elementGraphDataSize - (sizeof(ElementGraphData) - sizeof(ElementLevelData));
        this->level0.copyInto(&dst->level0, level0Size, allocator);
        if (this->toplevel > 0) {
            dst->others =
                (ElementLevelData *)allocator->callocate(levelDataSize * this->toplevel);
            if (dst->others == nullptr) {
                throw std::runtime_error("VecSim index low memory error");
            }
            for (size_t i = 0; i < this->toplevel; i++) {
                auto *src_ld = reinterpret_cast<const ElementLevelData *>(
                    reinterpret_cast<const char *>(this->others) + i * levelDataSize);
                auto *dst_ld = reinterpret_cast<ElementLevelData *>(
                    reinterpret_cast<char *>(dst->others) + i * levelDataSize);
                src_ld->copyInto(dst_ld, levelDataSize, allocator);
            }
        } else {
            dst->others = nullptr;
        }
    }
};
