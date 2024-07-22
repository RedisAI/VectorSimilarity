#pragma once

#include "abs_graph_data.h"

class IncomingEdges : private     vecsim_stl::vector<idType> ,
                      public absEdges {
public:
    IncomingEdges(std::shared_ptr<VecSimAllocator> allocator) :
        vecsim_stl::vector<idType>(allocator) {}

    void
    push(idType id) override {
        vecsim_stl::vector<idType>::push_back(id);
    }
    
    bool
    removeIdIfExists(idType element_id) override {
        auto it = std::find(begin(), end(), element_id);
        if (it != end()) {
            // Swap the last element with the current one (equivalent to removing the element id from
            // the list).
            *it = back();
            pop_back();
            return true;
        }
        return false;
    }

    virtual  void
    removeId(idType element_id) override {
        auto exists = removeIdIfExists(element_id);
        if (!exists)
            assert(0);
    }

    std::pair<size_t, const idType *> 
    Get() override {
        return {size(), data()};
    }
    virtual void
    Set(std::pair<size_t , const idType *> inp) override {
        resize(inp.first);
        memcpy(data(), inp.second, inp.first * sizeof(idType));
    } 

    void save(std::ofstream &output);
    void restore(std::ifstream &input);
};


class OutgoingEdges :  public absEdges {
public:
    OutgoingEdges() :num_links_(0) {}
    ~OutgoingEdges() = default;
    
    void
    push(idType id) override {
        links_[num_links_++] = id;
    }
    
    bool
    removeIdIfExists(idType element_id) override {
        for (size_t i = 0; i < num_links_; i++) {
            if (links_[i] == element_id) {
                // Swap the last element with the current one (equivalent to removing the element id from
                // the list).
                links_[i] = links_[num_links_-1];
                num_links_--;
                return true;
            }
        }
        return false;
    }

    virtual  void
    removeId(idType element_id) override {
        auto exists = removeIdIfExists(element_id);
        if (!exists)
            assert(0);
    }

    std::pair<size_t, const idType *>
    Get() {
        return {num_links_, links_};
    }
    virtual void
    Set(std::pair<size_t , const idType *> inp) override {
        num_links_ = inp.first;
        memcpy(links_, inp.second, inp.first * sizeof(idType));
    } 

    void save(std::ofstream &output);
    void restore(std::ifstream &input);
private:
    size_t num_links_=0;
    idType links_[];
};


class WriteBatch;

class VectorGraphData;
class LevelData;
class RamGraphData : public absGraphData  {
private:
	friend class absGraphData;
    RamGraphData(std::shared_ptr<VecSimAllocator> allocator,
                 size_t block_size,
                 size_t max_num_outgoing_links,
                 size_t vector_size_bytes,
                 size_t initial_capacity,
                 size_t vector_alignment);
    
    virtual ~RamGraphData() override {};


    // vector methods 

    virtual const char *
    getVectorByInternalId(idType internal_id) const override;
    
    virtual void
    multiGetVectors(const std::vector<idType> &,
                    std::vector<const char *> &results) const override;
    
    virtual idType
    pushVector(const void *vector_data,
               int max_level,
               const labelType  &label,
               WriteBatch *wb) override;

    // vectorMetaData methods
    
    const VectorMetaData &
    vectorMetaDataById(idType internal_id) const override {
        return idToMetaData_[internal_id];
    }
    
    VectorMetaData    &
    vectorMetaDataById(idType internal_id,
                       WriteBatch *) override {
        return idToMetaData_[internal_id];
    }

    // premanently delete the vector and the edges "free" the id 
    void
    deleteVectorAndEdges(idType internalId,
                         WriteBatch *wb) override {
		vectorMetaDataById(internalId, wb).mark(
			VectorMetaData::PERMANENT_DELETED);
	}

    
    // outgoing edges 
    virtual const absEdges &
    GetLevelOutgoingEdges(const graphNodeType &) const override;

    virtual absEdges &
    GetLevelOutgoingEdges(const graphNodeType &,
                          WriteBatch *) override;


    // inomming edges 
    // fetch incoming from the database
    virtual const absEdges &
    GetLevelIncomingEdges(const graphNodeType &) const override;
    virtual absEdges &
    GetLevelIncomingEdges(const graphNodeType &,
                          WriteBatch *) override;

    // May not fetch from the database support only
    // simple updates (add / delete target) operations 
    virtual absEdges &
    GetLevelVirtualIncomingEdges(const graphNodeType &id,
                                 WriteBatch *wb) override {
        // in mem implementation 
        return GetLevelIncomingEdges(id, wb);
    }

public:
    // helper methods needed by hnsw

    // get the first id that exists at level "level"
	// at or after the statingId
    virtual idType
    getVectorIdByLevel(short level,
					   idType startingId = 0) const = 0;

    // get a permanent deleted entry (at or after start point)
	// to be used by the GC 
    virtual idType
    getGarbadgeCollectionTarget(idType startingId = 0) const = 0;
    
    virtual void
    shrinkToFit() override;

public:
    // new and commit wrire batch are not supported (all writes are inplace) 
    WriteBatch *
    newWriteBatch() override {
        return nullptr;
    }

    void
    CommitWriteBatch(WriteBatch *) override {
        return;
    }

public:
    virtual void
    save(std::ofstream &output) const override;
    virtual void
    restore(std::ifstream &input) override;
    
private:
    VectorGraphData *
    getGraphDataByInternalId(idType internal_id) const;
    
    void growByBlock();
    void shrinkByBlock();
    
private:
    vecsim_stl::vector<DataBlock> vectorBlocks_;
    vecsim_stl::vector<DataBlock> graphDataBlocks_;
    vecsim_stl::vector<VectorMetaData> idToMetaData_;
    std::shared_ptr<VecSimAllocator> allocator_;    
    const size_t block_size_;
    const size_t vector_size_bytes_;
    const size_t vector_alignment_;
    
    
};
