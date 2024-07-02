
#pragma once

#include <cassert>
#include "VecSim/utils/vecsim_stl.h"

template <typename DistType>
using candidatesList = vecsim_stl::vector<std::pair<DistType, idType>>;

typedef uint16_t linkListSize;

// Helper method that swaps the last element in the ids list with the given one (equivalent to
// removing the given element id from the list).
bool removeIdFromList(vecsim_stl::vector<idType> &element_ids_list, idType element_id);

struct LevelData {
    // A list of ids that are pointing to the node where each edge is *unidirectional*
    vecsim_stl::vector<idType> *incomingEdges;
    // Total size of incoming links to the node (both uni and bi directinal).
    linkListSize totalIncomingLinks;
    linkListSize numLinks;
    // Flexible array member - https://en.wikipedia.org/wiki/Flexible_array_member
    // Using this trick, we can have the links list as part of the LevelData struct, and avoid
    // the need to dereference a pointer to get to the links list.
    // We have to calculate the size of the struct manually, as `sizeof(LevelData)` will not include
    // this member. We do so in the constructor of the index, under the name `levelDataSize` (and
    // `elementGraphDataSize`). Notice that this member must be the last member of the struct and
    // all nesting structs.
    idType links[];

    explicit LevelData(std::shared_ptr<VecSimAllocator> allocator)
        : incomingEdges(new (allocator) vecsim_stl::vector<idType>(allocator)),
          totalIncomingLinks(0), numLinks(0) {}

    linkListSize getNumLinks() const { return this->numLinks; }
    idType getLinkAtPos(size_t pos) const {
        assert(pos < numLinks);
        return this->links[pos];
    }
    const vecsim_stl::vector<idType> &getIncomingEdges() const { return *incomingEdges; }
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
    void newIncomingUnidirectionalEdge(idType node_id) { this->incomingEdges->push_back(node_id); }
    bool removeIncomingUnidirectionalEdgeIfExists(idType node_id) {
        return removeIdFromList(*this->incomingEdges, node_id);
    }
    void increaseTotalIncomingEdgesNum() { this->totalIncomingLinks++; }
    void decreaseTotalIncomingEdgesNum() { this->totalIncomingLinks--; }
    void swapNodeIdInIncomingEdges(idType id_before, idType id_after) {
        auto it = std::find(this->incomingEdges->begin(), this->incomingEdges->end(), id_before);
        // This should always succeed
        assert(it != this->incomingEdges->end());
        *it = id_after;
    }
};

struct ElementGraphData {
    size_t toplevel;
    std::mutex neighborsGuard;
    LevelData *others;
    LevelData level0;

    ElementGraphData(size_t maxLevel, size_t high_level_size,
                     std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0(allocator) {
        if (toplevel > 0) {
            others = (LevelData *)allocator->callocate(high_level_size * toplevel);
            if (others == nullptr) {
                throw std::runtime_error("VecSim index low memory error");
            }
            for (size_t i = 0; i < maxLevel; i++) {
                new ((char *)others + i * high_level_size) LevelData(allocator);
            }
        }
    }
    ~ElementGraphData() = delete; // Should be destroyed using `destroyGraphData`
};
