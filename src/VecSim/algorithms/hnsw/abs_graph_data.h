#pragma once

#include <deque>
#include <memory>
#include <cassert>
#include <climits>
#include <queue>
#include <random>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <sys/resource.h>
#include <fstream>
#include <shared_mutex>

#include "visited_nodes_handler.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/data_block.h"
#include "VecSim/utils/vecsim_results_container.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/tombstone_interface.h"

#ifdef BUILD_TESTS
#include "hnsw_serialization_utils.h"
#include "VecSim/utils/serializer.h"
#endif

using std::pair;
using graphNodeType = pair<idType, ushort>; // represented as: (element_id, level)



class absEdges  {
public:
    absEdges();
    virtual ~absEdges();
    
    virtual void push(idType id) = 0;
    
    virtual  bool removeIdIfExists(idType element_id) = 0;
    virtual  void removeId(idType element_id) = 0;
    
    virtual std::pair<size_t, const idType *> Get() = 0;
    virtual void Set(std::pair<size_t, const idType *> inp) = 0;
    
    virtual void save(std::ofstream &output) ;
    virtual void restore(std::ifstream &input);
};
        

// vector metadata contains  all the metadata of the vector;
// this is replacing the id->metadata table and the element graph data
// 

struct VectorMetaData 
{
     enum Flags {
         DELETE_MARK = 0x1, // element is logically deleted, but still exists in the graph
         IN_PROCESS = 0x2,  // element is being inserted into the graph
         PERMANENT_DELETED = 0x4,  // element no longer in the graph
     }; 
    VectorMetaData(const labelType &label, uint8_t max_level) :
        label_(label), max_level_(max_level), flags_(0) {}
	
    VectorMetaData(const VectorMetaData &src) :
        label_(src.label_), max_level_(src.max_level_)
		{flags_ = char(src.flags_);}

    // mark methods 
    void mark(Flags flag) {
        flags_ |= flag;
    }        
    void unmark(Flags flag) {
        flags_ &= ~flag;
    }
    bool ismarked(Flags flag)  const  {
        return flags_ & flag;
    }
    
    labelType label_;
    uint8_t  max_level_;
    std::atomic<uint8_t> flags_ = 0;    
    std::mutex NodeGuard; 
};


class WriteBatch;
class absGraphData  {
public:
    absGraphData() {}    
    virtual ~absGraphData() {};    

    // vector methods 
    virtual const char *
    getVectorByInternalId(idType internal_id) const = 0;
    
    virtual void
    multiGetVectors(const std::vector<idType> &,
                    std::vector<const char *> &results) const = 0;
    
    virtual idType
    pushVector(const void *vector_data,
               int max_level,
               const labelType  &label,
               WriteBatch *wb) = 0;

    // premanently delete the vector and the edges "free" the id 
    virtual void
    deleteVectorAndEdges(idType internalId,
                         WriteBatch *wb) = 0;


    // vectorMetaData methods
    virtual const VectorMetaData &
    vectorMetaDataById(idType internal_id) const = 0;
        
    
    virtual VectorMetaData    &
    vectorMetaDataById(idType internal_id,
                       WriteBatch *wb);
    



    
    // outgoing edges 
    virtual const absEdges &
    GetLevelOutgoingEdges(const graphNodeType &) const = 0;

    virtual absEdges &
    GetLevelOutgoingEdges(const graphNodeType &,
                          WriteBatch *) = 0;


    // inomming edges 
    // fetch incoming from the database
    virtual const absEdges &
    GetLevelIncomingEdges(const graphNodeType &) const = 0;
    virtual absEdges &
    GetLevelIncomingEdges(const graphNodeType &,
                          WriteBatch *) = 0;

    // support only simple updates (add / delete target) operations
    // may not fetch the data from the database
    virtual absEdges &
    GetLevelVirtualIncomingEdges(const graphNodeType &id,
                                 WriteBatch *) = 0;
    // helper methods
    
    // scan the database for the first node after starting id that exist at level
    virtual idType
    getVectorIdByLevel(short level,
                    idType startingId) const = 0;

    // get a pair of candidates to swap for the gc
    // first is a location that is permanent deleted
    // second is a location that is valid
    // start points is the last pair returned in the prev scan 
    virtual idType
    getGarbadgeCollectionTarget(idType startPoint) const = 0;

    // new and commit wrire batch 
    virtual WriteBatch *newWriteBatch() = 0;
    virtual void CommitWriteBatch(WriteBatch *wb) = 0;

    
    virtual void shrinkToFit() = 0;

public:
    virtual void save(std::ofstream &output) const = 0;
    virtual void restore(std::ifstream &input) = 0;

	static absGraphData *
	NewRamGraphData(std::shared_ptr<VecSimAllocator> allocator,
					size_t block_size,
					size_t max_num_outgoing_links,
					size_t vector_size_bytes,
					size_t initial_capacity,
					size_t vector_alignment);

	static absGraphData *
	NewRamWBGraphData(std::shared_ptr<VecSimAllocator> allocator,
					  size_t block_size,
					  size_t max_num_outgoing_links,
					  size_t vector_size_bytes,
					  size_t initial_capacity,
					  size_t vector_alignment);

	static absGraphData *
	NewDBGraphData(std::shared_ptr<VecSimAllocator> allocator,
				   std::string db_path);


protected:
    
};

