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

using graphNodeType = std::pair<idType, ushort>; // represented as: (element_id, level)
#include "abs_data_store.h"


static inline void  
buildKey(const graphNodeType gn, std::string &ret) {
	ret.resize(16);;
	sprintf(ret.data(),"%8.8u%8.8u",gn.second,gn.first);
}

static inline graphNodeType
extractKey(const char *key) {
	unsigned int level, id;
	sscanf(key,"%8u%8u",&level,&id);
	return 	graphNodeType({id, level});
}

class ObjectChange
{
public:
	virtual ~ObjectChange() {} ;
	virtual void GetStageParams(StageParams &) = 0;
	virtual void Apply(std::vector<idType> &cur_value) = 0;
	virtual void Apply(size_t &size, idType *cur_value) = 0;
	static ObjectChange *NewObjectChangeFromString(const char *data);
};




class DirtyObject {
public:

	DirtyObject() = default;
	DirtyObject(const std::vector<idType>  &from) :
		was_fetched_(true), object_value_(from) {
	}
	virtual ~DirtyObject() {};

	// append the operation that are pending for stages 
	virtual void
	AppendStagesOp(const graphNodeType &gn,
				   std::list<StageParams> &stageParmsList)  = 0;

	void ApplyChanges() {
		assert(was_fetched_);
		for (auto ch : all_changes_) {
			ch->Apply(object_value_);
			is_dirty_ = true;
			delete ch;
		}
		all_changes_.clear();
	}
			
	const std::vector<idType> &Get() {
		assert(was_fetched_);		
		ApplyChanges();
		return object_value_;
	}
	
	void SetFetchedData(const std::vector<idType> &data) {
		object_value_ = data;
		was_fetched_ = true;
		is_dirty_ = false;
	}
			
	void Set(const std::vector<idType> &data) {
		object_value_ = data;
		was_fetched_ = true;
		is_dirty_ = true;
		for (auto ch : all_changes_) {
			delete ch;
		}
		all_changes_.clear(); 
	}
	
	void  addChange(ObjectChange *ch) {
		all_changes_.push_back(ch);
	}
	bool fetched() const {return was_fetched_;}
	bool &fetched() {return was_fetched_;}
	virtual CF_ID GetCfId() = 0;
			
	

protected:
	std::list<ObjectChange *> all_changes_;
	// data = null is not enough as it may be a new object
	bool was_fetched_ = false;
	bool is_dirty_ = false; 
	std::vector<idType>  object_value_;
};

// a container of dirty objects 
class WriteBatch {
public:
	void Register(std::string &key, DirtyObject  *obj) {
		auto cf_id = obj->GetCfId();
		auto true_key = key + std::string(1, (char)cf_id);
		auto iter = dirty_objects.find(true_key);
		if (iter == dirty_objects.end()) {
			dirty_objects.insert({true_key, obj});
		}
	}
	
	
	DirtyObject *find(std::string &key, int cf_id) const {
		auto true_key = key + std::string(1, (char)cf_id);
		auto const &iter = dirty_objects.find(true_key);
		if (iter == dirty_objects.end()) {
			return nullptr;
		} else {
			return iter->second;		   
		}
	}

	const std::unordered_map<std::string, DirtyObject*>::const_iterator begin() const
		{ return dirty_objects.begin(); }
	const std::unordered_map<std::string, DirtyObject*>::const_iterator end() const
		{ return dirty_objects.end(); }
	
	
		
private:
	std::unordered_map<std::string, DirtyObject*> dirty_objects;
};

	

	



