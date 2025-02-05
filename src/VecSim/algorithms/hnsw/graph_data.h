
#pragma once

#include <cassert>
#include <algorithm>
#include <mutex>

#include "VecSim/utils/vec_utils.h"
#include "VecSim/containers/mapped_mem.h"

template <typename DistType>
using candidatesList = vecsim_stl::vector<std::pair<DistType, idType>>;

typedef uint16_t linkListSize;

namespace graphDataUtils {
static size_t levelIdx(size_t level) {
  return level - 1; // no need to store id's offset for level 0, it's sequential.
}
}
struct ElementInMemoryData {
    ElementInMemoryData(vecsim_stl::vector<idType> offsets, std::shared_ptr<VecSimAllocator> allocator) :
    offsetIdxAtLevel(offsets, allocator), incomingUnidirectionalEdges(offsetIdxAtLevel.size() + 1, allocator)
    // ,offsetIdxAtLevel(new(allocator) vecsim_stl::vector<idType>(offsets.size(), allocator))
    {
        for (auto &inc_edges_ptr : incomingUnidirectionalEdges) {
            inc_edges_ptr = new(allocator) vecsim_stl::vector<idType>(allocator);
        }
        // (*offsetIdxAtLevel) = offsets;
    }
    ElementInMemoryData(ElementInMemoryData&& other) noexcept
        :offsetIdxAtLevel(std::move(other.offsetIdxAtLevel))  ,
          incomingUnidirectionalEdges(std::move(other.incomingUnidirectionalEdges))
          {}

    mutable std::mutex neighborsGuard;
    // offsetAtLevel[i] = relative offset of the element data in i = graphDataUtils::levelIdx(level).
    vecsim_stl::vector<idType> offsetIdxAtLevel; // offsets of the element at each level > 0.
    vecsim_stl::vector<vecsim_stl::vector<idType> *> incomingUnidirectionalEdges;

    ~ElementInMemoryData() {
        for (auto &inc_edges_ptr : incomingUnidirectionalEdges) {
            delete inc_edges_ptr;
        }
    }
    size_t getOffsetAtLevel(size_t level) const {
        return offsetIdxAtLevel.at(graphDataUtils::levelIdx(level));
    }

    size_t getMaxlevel() const { return offsetIdxAtLevel.size(); }
};

/******* Disk structs *******/

struct LevelsMappedMemContainer { // TODO: separate struct for level 0
    LevelsMappedMemContainer(size_t elementDataSize, std::shared_ptr<VecSimAllocator> allocator, size_t cap = 0, bool is_level0 = false)
        : mappedMems(cap, allocator), DataSize(elementDataSize)  {
        is_level0 ? offsetLevel = 0 : offsetLevel = 1;
    }

    void destroy(size_t elementDataSize, size_t block_size_bytes) {
        for (size_t i = 0; i < mappedMems.size(); i++) {
            mappedMems[i].destroy(elementDataSize, block_size_bytes);
        }
    }

    // Return data of the element at offset_id in level
    char *getOffsetIdDataByLevel(idType offset_id, size_t level) const {
        return mappedMems[level - offsetLevel].mapped_addr + offset_id * DataSize;
    }

    // Append element to the end of 0, 1, 2...elem_max_level mappedMems
    // Returns the offset index of the new element
    void appendElementUpToLevel(const void *element, size_t element_size_bytes, size_t elem_max_level = 0) {
        for (size_t level = offsetLevel; level <= elem_max_level; level++) {
            mappedMems[level - offsetLevel].appendElement(element, element_size_bytes);
        }
    }

    size_t getElemCountByLevel(size_t level) const {
        return mappedMems[level - offsetLevel].get_elem_count();
    }

    size_t getLevelsCount() const { return mappedMems.size(); }

    void UpdateMaxLevel(size_t maxLevel) {
        if (getLevelsCount() < maxLevel) {
            mappedMems.resize(maxLevel);
        }
    }

    bool growByBlockUpTolevel(size_t elementDataSize, size_t block_size_bytes, size_t maxLevel) {
        bool is_resized = false;
        for (size_t i = 0; i < mappedMems.size(); i++) {
            is_resized |= mappedMems[i].growByBlock(elementDataSize, block_size_bytes);
        }
        return is_resized;
    }
    vecsim_stl::vector<MappedMem> mappedMems;
    size_t DataSize;
    size_t offsetLevel;
};

struct DiskElementMetaData {
    DiskElementMetaData(size_t toplevel)
        : toplevel(toplevel) {}
    const size_t toplevel;
};

// Used to read data from disk
struct ElementLevelData {
    // A list of ids that are pointing to the node where each edge is *unidirectional*
    vecsim_stl::vector<idType> *incomingUnidirectionalEdges;

    // Cache the currlinks to avoid reading from disk
    linkListSize currLinks;

    // Pointer to disk mapped memory
    // {linkListSize numLinks, idType link0, idType link1, ...}
    char *linksData;

    // explicit ElementLevelData(std::shared_ptr<VecSimAllocator> allocator)
    //     : incomingUnidirectionalEdges(new(allocator) vecsim_stl::vector<idType>(allocator)),
    //       numLinks(0) {}


    ElementLevelData() = default;
    explicit ElementLevelData(vecsim_stl::vector<idType> *incEdgesPtr, char *linksMappedMem)
        : incomingUnidirectionalEdges(incEdgesPtr), currLinks(*((linkListSize *)linksMappedMem)), linksData(linksMappedMem) {}

    linkListSize getNumLinks() const { return this->currLinks; }

    idType *getLinksArray() const {
        return (idType *)((linkListSize *)linksData + 1); // skip numLinks
    }
    idType getLinkAtPos(size_t pos) const {
        assert(pos < currLinks);
        return getLinksArray()[pos];
    }
    const vecsim_stl::vector<idType> &getIncomingEdges() const {
        return *incomingUnidirectionalEdges;
    }
    std::vector<idType> copyLinks() {
        std::vector<idType> links_copy;
        idType *links = getLinksArray();
        links_copy.assign(links, links + currLinks);
        return links_copy;
    }
    // Sets the outgoing links of the current element.
    // Assumes that the object has the capacity to hold all the links.
    void setLinks(vecsim_stl::vector<idType> &links) {
        currLinks = links.size();
        *(linkListSize *)linksData = currLinks;
        memcpy(getLinksArray(), links.data(), currLinks * sizeof(idType));
    }
    template <typename DistType>
    void setLinks(candidatesList<DistType> &cand_links) {
        currLinks = cand_links.size();
        *(linkListSize *)linksData = currLinks;
        idType *links = getLinksArray();
        for (auto &link : cand_links) {
            links = link.second;
            links++;
        }
    }
    void popLink() { this->currLinks--; *(linkListSize *)linksData = currLinks;}
    void setNumLinks(linkListSize num) { this->currLinks = num; *(linkListSize *)linksData = currLinks;}
    void setLinkAtPos(size_t pos, idType node_id) { this->getLinksArray()[pos] = node_id; }
    void appendLink(idType node_id) {
        this->getLinksArray()[this->currLinks++] = node_id;
        *(linkListSize *)linksData = currLinks;
    }
    void removeLink(idType node_id) {
        idType *links = getLinksArray();
        size_t i = 0;
        for (; i < currLinks; i++) {
            if (links[i] == node_id) {
                links[i] = links[currLinks - 1];
                break;
            }
        }
        assert(i < currLinks && "Corruption in HNSW index"); // node_id not found - error
        popLink();
    }
    void newIncomingUnidirectionalEdge(idType node_id) {
        this->incomingUnidirectionalEdges->push_back(node_id);
    }
    bool removeIncomingUnidirectionalEdgeIfExists(idType node_id) {
        return this->incomingUnidirectionalEdges->remove(node_id);
    }
    void swapNodeIdInIncomingEdges(idType id_before, idType id_after) {
        auto it = std::find(this->incomingUnidirectionalEdges->begin(),
                            this->incomingUnidirectionalEdges->end(), id_before);
        // This should always succeed
        assert(it != this->incomingUnidirectionalEdges->end());
        *it = id_after;
    }
};

struct DiskElementGraphDataCopy {
    size_t toplevel; // TODO: redundant ?
    vecsim_stl::vector<ElementLevelData> levelsData;
    mutable std::mutex* neighborsGuard;

    DiskElementGraphDataCopy(size_t toplevel,
                         const vecsim_stl::vector<ElementLevelData>& levelsData,
                         std::mutex& neighborsGuard)
        : toplevel(toplevel), levelsData(levelsData), neighborsGuard(&neighborsGuard) {}

    const ElementLevelData &getElementLevelData(size_t level) const {
        assert(level <= toplevel);
        return levelsData[level];
    }

    ElementLevelData &getElementLevelData(size_t level) {
        assert(level <= toplevel);
        return levelsData[level];
    }

    void lockNodeLinks() const {
        (neighborsGuard)->lock();
    }

    void unlockNodeLinks() const {
        (neighborsGuard)->unlock();
    }

    void destroy() {
        for (size_t i = 0; i < levelsData.size(); i++) {
            delete levelsData[i].incomingUnidirectionalEdges;
        }
    }
};
struct GraphData : public VecsimBaseObject {
   //LevelsMappedMemContainer(size_t elementDataSize, std::shared_ptr<VecSimAllocator> allocator, size_t cap = 0, bool is_level0 = false)
    size_t level0DataSize; // size of each element in level0
    size_t levelDataSize; // size of each element in levels > 0
    LevelsMappedMemContainer MetaDatasAndLevel0; // A file for all elements' meta data + level0
    LevelsMappedMemContainer levelsData; // File for each level
    vecsim_stl::vector<ElementInMemoryData> InMemoryElementsData; // ElementInMemoryData elements
    size_t level0DatablockSizeBytes; // page size of the system
    size_t levelsDatablockSizeBytes; // page size of the system

    GraphData(size_t M0, size_t M, std::shared_ptr<VecSimAllocator> allocator) : VecsimBaseObject(allocator),
        // each element contains: DiskElementMetaData, numLinks, link0, link1, ... link(M0-1)
     level0DataSize(sizeof(DiskElementMetaData) + sizeof(linkListSize) + M0 * sizeof(idType)),
        // each element contains: numLinks, link0, link1, ... link(M-1)
     levelDataSize(sizeof(linkListSize) + M * sizeof(idType)),
     MetaDatasAndLevel0(level0DataSize, allocator, 1, true),
     levelsData(levelDataSize, allocator),
     InMemoryElementsData(allocator) {

        size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));

        // let one data block be at least 1 page
        level0DatablockSizeBytes = MAX(pageSize, level0DataSize * DEFAULT_BLOCK_SIZE);
        levelsDatablockSizeBytes = MAX(pageSize, levelDataSize * DEFAULT_BLOCK_SIZE);
    };

    ~GraphData() {
        MetaDatasAndLevel0.destroy(level0DataSize, level0DatablockSizeBytes);
        levelsData.destroy(levelDataSize, levelsDatablockSizeBytes);
    }

    ElementLevelData getElementLevelData(idType internal_id, size_t level) const {
        const ElementInMemoryData &inMemoryData = InMemoryElementsData[internal_id];
        vecsim_stl::vector<idType> *inc_edges = inMemoryData.incomingUnidirectionalEdges[level];
        char *linksData = nullptr;
        if (level == 0) {
            linksData = MetaDatasAndLevel0.getOffsetIdDataByLevel(internal_id, level) + sizeof(DiskElementMetaData);
        } else {
            size_t offsetAtlevel = inMemoryData.getOffsetAtLevel(level);
            linksData = levelsData.getOffsetIdDataByLevel(offsetAtlevel, level);
        }
        return ElementLevelData(inc_edges, linksData);
    }

    void UpdateMaxLevel(size_t newMaxLevel) {
        levelsData.UpdateMaxLevel(newMaxLevel);
    }

    void appendElement(size_t toplevel, labelType label, size_t id) {
        // emplace space in levels data if needed for the new element
        growByBlock(toplevel);

        // Add the in memory data
        vecsim_stl::vector<idType> offsets(toplevel, this->allocator);
        for (size_t i = 1; i <= toplevel; i++) {
            idType elem_index_at_level = levelsData.getElemCountByLevel(i);
            offsets[i - 1] = elem_index_at_level;
        }

        InMemoryElementsData.emplace_back(offsets, this->allocator);


        // create ElementLevel0Data
        char level0Data[this->level0DataSize] = {0};

        DiskElementMetaData metadata(toplevel);
        memcpy(level0Data, &metadata, sizeof(DiskElementMetaData));
        MetaDatasAndLevel0.appendElementUpToLevel(level0Data, this->level0DataSize, 0);

        // add to all level up to toplevel
        char levelData[this->levelDataSize] = {0};
        levelsData.appendElementUpToLevel(levelData, this->levelDataSize, toplevel);
    }

    size_t getElemMaxLevel(idType id) {
        return InMemoryElementsData[id].getMaxlevel();
    }

    void removeElement(size_t id) {
        // TODO: make sure we freed the element memory before overriding it
        // TODO: make sure inMemoryData is handled properly in hnsw.h


        // override the element data with the last element data
        // Do the same for the rest of the levels'
        // size_t elem_max_level = getElemMaxLevel(id);
        // for (size_t i = 0; i < elem_max_level; i++) {
        //     idType last_element_internal_id = levelsData[i]->last_elem_id;
        //     char *last_elem_level_file_ptr = getLevelDataByInternalId(i, last_element_internal_id);
        //     char *elem_level_file_ptr = getLevelDataByInternalId(element_internal_id);
        //     memcpy(elem_level_file_ptr, last_elem_level_file_ptr, this->elementlevelDataSize);
        // }

        // // create ElementInMemoryData
        // InMemoryElementsData.addElement(inMemoryData, id);

    }

    void growByBlock(size_t maxLevel) {
        MetaDatasAndLevel0.growByBlockUpTolevel(level0DataSize, level0DatablockSizeBytes, 0);
        if (maxLevel > 0) {
            levelsData.UpdateMaxLevel(maxLevel);
            levelsData.growByBlockUpTolevel(levelDataSize, levelsDatablockSizeBytes, maxLevel);
        }
        if ((InMemoryElementsData.size() % DEFAULT_BLOCK_SIZE) == 0) {
            InMemoryElementsData.reserve(InMemoryElementsData.size() + DEFAULT_BLOCK_SIZE);
        }
    }

    DiskElementGraphDataCopy getGraphDataByInternalId(idType internal_id) const {
        const ElementInMemoryData &elemInMemData = InMemoryElementsData[internal_id];
        size_t toplevel = elemInMemData.getMaxlevel();
        vecsim_stl::vector<ElementLevelData> levelsData(toplevel + 1, this->allocator);
        for (size_t level = 0; level <= toplevel; level++) {
            levelsData[level]= getElementLevelData(internal_id, level);
        }

        return DiskElementGraphDataCopy(toplevel, levelsData, elemInMemData.neighborsGuard);
    }

    void lockNodeLinks(idType internal_id) const {
        InMemoryElementsData[internal_id].neighborsGuard.lock();
    }

    void unlockNodeLinks(idType internal_id) const {
        InMemoryElementsData[internal_id].neighborsGuard.unlock();
    }
};




// struct ElementGraphData {
//     size_t toplevel;
//     std::mutex neighborsGuard;
//     ElementLevelData *others;
//     ElementLevelData level0;

//     ElementGraphData(size_t maxLevel, size_t high_level_size,
//                      std::shared_ptr<VecSimAllocator> allocator)
//         : toplevel(maxLevel), others(nullptr), level0(allocator) {
//         if (toplevel > 0) {
//             others = (ElementLevelData *)allocator->callocate(high_level_size * toplevel);
//             if (others == nullptr) {
//                 throw std::runtime_error("VecSim index low memory error");
//             }
//             for (size_t i = 0; i < maxLevel; i++) {
//                 new ((char *)others + i * high_level_size) ElementLevelData(allocator);
//             }
//         }
//     }
//     ~ElementGraphData() = delete; // should be destroyed using `destroy'

//     void destroy(size_t levelDataSize, std::shared_ptr<VecSimAllocator> allocator) {
//         delete this->level0.incomingUnidirectionalEdges;
//         ElementLevelData *cur_ld = this->others;
//         for (size_t i = 0; i < this->toplevel; i++) {
//             delete cur_ld->incomingUnidirectionalEdges;
//             cur_ld = reinterpret_cast<ElementLevelData *>(reinterpret_cast<char *>(cur_ld) +
//                                                           levelDataSize);
//         }
//         allocator->free_allocation(this->others);
//     }
//     ElementLevelData &getElementLevelData(size_t level, size_t levelDataSize) {
//         assert(level <= this->toplevel);
//         if (level == 0) {
//             return this->level0;
//         }
//         return *reinterpret_cast<ElementLevelData *>(reinterpret_cast<char *>(this->others) +
//                                                      (level - 1) * levelDataSize);
//     }
// };
