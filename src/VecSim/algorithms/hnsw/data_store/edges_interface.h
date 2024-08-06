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
#include "../hnsw_serialization_utils.h"
#include "VecSim/utils/serializer.h"
#endif

using std::pair;
using graphNodeType = pair<idType, ushort>; // represented as: (element_id, level)


class WriteBatch;
class EdgesDataStore;
EdgesDataStore *NewRamDataStore(std::shared_ptr<VecSimAllocator> allocator,
								size_t block_size,
								size_t max_num_outgoing_links,
								size_t initial_capacity);

EdgesDataStore *NewSpeedbDataStore(std::shared_ptr<VecSimAllocator> allocator,
								   const char *dbPath);

// edges interface provide encapsulation of accesses to the graph edges
// edges interface support also write batch
// if the user is using a wb all her changes will be aggregaded and apply at once on the datastor
// all the methods recive a graph node type to define the object 


class EdgesInterface {
public:
	EdgesInterface(std::shared_ptr<VecSimAllocator> allocator,
				  EdgesDataStore *ds) :		
		allocator_(allocator),
		ds_(ds) {
	}
    ~EdgesInterface() {}; 
    // outgoing edges 
    const std::vector<idType> 
    GetOutgoingEdges(const graphNodeType &,
					 WriteBatch *) const;
		
	
	// add one target
	void AddOutgoingTarget(const graphNodeType &gn,
						   idType , 
						   WriteBatch *);
		
    // delete one target
	void DeleteOutgoingTarget(const graphNodeType &,
							  idType target,
							  WriteBatch *);

	// set all the targets
	void SetOutgoingAllTargets(const graphNodeType &,
							   const std::vector<idType> &,
							   WriteBatch *);
	
	
    // incoming edges 
	const std::vector<idType> 
    GetIncomingEdges(const graphNodeType &,
					 WriteBatch *) const ;
	// add one target
	void AddIncomingTarget(const graphNodeType &gn,
						   idType , 
						   WriteBatch *);
		
    // delete one target
	void DeleteIncomingTarget(const graphNodeType &,
							  idType target,
							  WriteBatch *);

	// set all the targets
	// setting with an empty val equal to delete the edges on this level  
	void SetIncomingAllTargets(const graphNodeType &,
							   const std::vector<idType> &,
							   WriteBatch *);
	
    // delete all the edges (outgoing and incoming) at all the levels 

	void DeleteNodeEdges(const idType id);
	WriteBatch *newWriteBatch();
	void CommitWriteBatch(WriteBatch *);

	void Flush();
private:
	std::shared_ptr<VecSimAllocator> allocator_;	
	EdgesDataStore *ds_;
	
};



	


