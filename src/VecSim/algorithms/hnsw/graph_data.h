
#pragma once

#include <cassert>
#include <algorithm>
#include <mutex>

#include "VecSim/utils/vec_utils.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

template <typename DistType>
using candidatesList = vecsim_stl::vector<std::pair<DistType, idType>>;

typedef uint16_t linkListSize;

struct ElementInMemoryData {
    mutable std::mutex neighborsGuard;
    size_t topLevel;

    ElementInMemoryData(size_t topLevel = 0) : topLevel(topLevel) {}
    ElementInMemoryData(const ElementInMemoryData &other) = delete;
    ElementInMemoryData(ElementInMemoryData &&other) noexcept : topLevel(other.topLevel) {}
    ElementInMemoryData &operator=(const ElementInMemoryData &other) = delete;
    ElementInMemoryData &operator=(ElementInMemoryData &&other) noexcept {
        topLevel = other.topLevel;
        return *this;
    }

    ~ElementInMemoryData() = default;

    size_t getMaxlevel() const { return topLevel; }
};

/******* Disk structs *******/

struct GraphData; // Forward declaration

// Used to read data from disk
struct ElementLevelData {
private:
    const GraphData *graph;
    std::string key; // key for the element (id:level)
    linkListSize numLinks;
    linkListSize cap;
    bool dirty;
    idType *links;

public:
    ElementLevelData(const GraphData *graph, std::string &&key, std::string &&value,
                     linkListSize cap)
        : graph(graph), key(key), cap(cap), dirty(false), links(new idType[cap]) {
        memcpy(links, value.data(), value.size());
        numLinks = value.size() / sizeof(idType);
        assert(numLinks <= cap);
    }
    inline ~ElementLevelData();

    linkListSize getNumLinks() const { return this->numLinks; }

    idType getLinkAtPos(size_t pos) const {
        assert(pos < numLinks);
        return links[pos];
    }
    std::vector<idType> copyLinks() { return std::vector<idType>(links, links + numLinks); }
    // Sets the outgoing links of the current element.
    // Assumes that the object has the capacity to hold all the links.
    void setLinks(vecsim_stl::vector<idType> &newLinks) {
        dirty = true;
        numLinks = newLinks.size();
        memcpy(links, newLinks.data(), numLinks * sizeof(idType));
    }
    template <typename DistType>
    void setLinks(candidatesList<DistType> &cand_links) {
        dirty = true;
        numLinks = cand_links.size();
        size_t i = 0;
        for (auto &link : cand_links) {
            links[i++] = link.second;
        }
    }
    void popLink() {
        dirty = true;
        this->numLinks--;
    }
    void setNumLinks(linkListSize num) {
        dirty = true;
        this->numLinks = num;
    }
    void setLinkAtPos(size_t pos, idType node_id) {
        assert(pos < numLinks);
        dirty = true;
        this->links[pos] = node_id;
    }
    void appendLink(idType node_id) {
        dirty = true;
        assert(numLinks < cap);
        this->links[numLinks++] = node_id;
    }
    void removeLink(idType node_id) {
        dirty = true;
        size_t i = 0;
        for (; i < numLinks; i++) {
            if (links[i] == node_id) {
                links[i] = links[numLinks - 1];
                break;
            }
        }
        assert(i < numLinks && "Corruption in HNSW index"); // node_id not found - error
        popLink();
    }
};

struct GraphData : public VecsimBaseObject {
    std::shared_ptr<rocksdb::DB> db;                              // RocksDB instance
    std::unique_ptr<rocksdb::ColumnFamilyHandle> cf;              // ColumnFamilyHandle instance
    vecsim_stl::vector<ElementInMemoryData> InMemoryElementsData; // ElementInMemoryData elements
    size_t M0;                                                    // size of each element in level0
    size_t M;

    GraphData(size_t M0, size_t M, std::shared_ptr<VecSimAllocator> allocator)
        : VecsimBaseObject(allocator), InMemoryElementsData(allocator), M0(M0), M(M) {}
    void setDB(std::shared_ptr<rocksdb::DB> db_) {
        db = db_;
        rocksdb::ColumnFamilyOptions cf_options;
        rocksdb::ColumnFamilyHandle *cf_;
        rocksdb::Status status = db->CreateColumnFamily(cf_options, "graph", &cf_);
        if (!status.ok()) {
            throw std::runtime_error("VecSim create column family 'graph' error");
        }
        cf.reset(cf_);
    };

    ~GraphData() = default;

    ElementLevelData getElementLevelData(idType internal_id, size_t level) const {
        std::string key(std::to_string(internal_id) + ":" + std::to_string(level));
        std::string value;
        rocksdb::Status status = db->Get(rocksdb::ReadOptions(), cf.get(), key, &value);
        if (!status.ok() && !status.IsNotFound()) {
            throw std::runtime_error("VecSim get element level data error");
        }
        return ElementLevelData(this, std::move(key), std::move(value), level ? M : M0);
    }

    void appendElement(size_t toplevel, labelType label, size_t id) {
        // emplace space in levels data if needed for the new element
        assert(InMemoryElementsData.size() == id);
        InMemoryElementsData.emplace_back(toplevel);
    }

    size_t getElemMaxLevel(idType id) { return InMemoryElementsData[id].getMaxlevel(); }

    void removeElement(size_t id) {
        // TODO: make sure we freed the element memory before overriding it
        // TODO: make sure inMemoryData is handled properly in hnsw.h

        // override the element data with the last element data
        // Do the same for the rest of the levels'
        // size_t elem_max_level = getElemMaxLevel(id);
        // for (size_t i = 0; i < elem_max_level; i++) {
        //     idType last_element_internal_id = levelsData[i]->last_elem_id;
        //     char *last_elem_level_file_ptr = getLevelDataByInternalId(i,
        //     last_element_internal_id); char *elem_level_file_ptr =
        //     getLevelDataByInternalId(element_internal_id); memcpy(elem_level_file_ptr,
        //     last_elem_level_file_ptr, this->elementlevelDataSize);
        // }

        // // create ElementInMemoryData
        // InMemoryElementsData.addElement(inMemoryData, id);
    }

    void growByBlock(size_t maxLevel) {
        // if (maxLevel > 0) {
        //     levelsData.UpdateMaxLevel(maxLevel);
        //     levelsData.growByBlockUpTolevel(levelDataSize, levelsDatablockSizeBytes, maxLevel);
        // }
        // if ((InMemoryElementsData.size() % DEFAULT_BLOCK_SIZE) == 0) {
        //     InMemoryElementsData.reserve(InMemoryElementsData.size() + DEFAULT_BLOCK_SIZE);
        // }
    }

    void lockNodeLinks(idType internal_id) const {
        InMemoryElementsData[internal_id].neighborsGuard.lock();
    }

    void unlockNodeLinks(idType internal_id) const {
        InMemoryElementsData[internal_id].neighborsGuard.unlock();
    }
};

inline ElementLevelData::~ElementLevelData() {
    if (dirty) {
        rocksdb::Slice value((char *)links, numLinks * sizeof(idType));
        graph->db->Put(rocksdb::WriteOptions(), graph->cf.get(), key, value);
    }
    delete[] links;
}
