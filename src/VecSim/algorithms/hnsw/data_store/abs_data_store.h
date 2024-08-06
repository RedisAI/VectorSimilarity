#pragma once
#include <list>
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

#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/utils/data_block.h"
#include "VecSim/utils/vecsim_results_container.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/tombstone_interface.h"


using graphNodeType = std::pair<idType, ushort>; // represented as: (element_id, level)
enum CF_ID {
	OUTGOING_CF = 1,
	INCOMING_CF = 2,
	LAST_CF = 3 // do not use (used only for indexing)	
};
struct StageParams
{
	enum Operation {
		WRITE_DATA,
		UPDATE_DATA,
		DELETE_DATA
		
	};
	CF_ID cf_id;
	Operation op;
	graphNodeType original_key;
	std::string key;
	std::string data;
};

struct GetParams
{
	CF_ID cf_id;
	graphNodeType original_key;
	std::string key;
	std::vector<idType> data;
};

class EdgesDataStore {
public:
	virtual void Get(std::list<GetParams> &) = 0;
	virtual void Put(const std::list<StageParams> &) = 0;
	virtual void Flush() {};
}; 
	
