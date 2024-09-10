#include "edges_interface.h"
#include "write_batch.h"
#include "edges.h"
#include "abs_data_store.h"


ObjectChange *ObjectChange::NewObjectChangeFromString(const char *data) {
	idType target = *(idType *) (data + 1);
	if (data[0] == 'A') {
		return new AddTarget(target);
	} else 	if (data[0] == 'D') {
		return new DeleteTarget(target);
	} else {
		assert(0);
		return nullptr;
	}
}
		


static  std::vector<idType> executeGet(EdgesDataStore *ds,
										const graphNodeType &gn,
										bool outgoing_edges ) {
	std::list<GetParams> getParamsList;
	if (outgoing_edges) {
		OutgoingEdges::PrepareGetsParams(gn, getParamsList);
	} else {
		IncomingEdges::PrepareGetsParams(gn, getParamsList);
	}
	 
	ds->Get(getParamsList);
	auto &getP = getParamsList.front();
	assert(extractKey(getP.key.data()) == gn);
	return getP.data;
	
}



// outgoing edges 
const std::vector<idType> 
EdgesInterface::GetOutgoingEdges(const graphNodeType &gn,
								WriteBatch *wb) const {
	if (wb) {
		std::string key;
		buildKey(gn, key);
		auto ret = wb->find(key, OutgoingEdges::CfId() );
		if (!ret) {
			ret = new OutgoingEdges();
			wb->Register(key, ret);
		} 
		if (!ret->fetched()) {
			ret->SetFetchedData(executeGet(ds_, gn, true));
			
		} 
		ret->ApplyChanges();
		return ((OutgoingEdges *)ret)->Get();		
	} else {
		return executeGet(ds_, gn, true);
	}		
}
		
	
// add one target
void
EdgesInterface::AddOutgoingTarget(const graphNodeType &gn,
								 idType target, 
								 WriteBatch *wb) {
	std::string key;
	buildKey(gn, key);		
	if (wb) {
		auto outEdges = wb->find(key, OutgoingEdges::CfId());		
		if (!outEdges) {
			outEdges = new OutgoingEdges(); 
			wb->Register(key, outEdges);
		}
		outEdges->addChange(new AddTarget(target));
	} else {
		std::list<StageParams> stage_params_list;		
		StageParams p;
		p.cf_id = OutgoingEdges::CfId();
		p.key = key;
		p.original_key = gn;
		AddTarget(target).GetStageParams(p);
		stage_params_list.push_back(p);
		ds_->Put(stage_params_list);
		
	}
	
}
// delete one target
void
EdgesInterface::DeleteOutgoingTarget(const graphNodeType &gn,
									idType target,
									WriteBatch *wb) {
	std::string key;
	buildKey(gn, key);		
	if (wb) {
		auto outEdges = wb->find(key, OutgoingEdges::CfId());		
		if (!outEdges) {
			outEdges = new OutgoingEdges(); 
			wb->Register(key, outEdges);
		}
		outEdges->addChange(new DeleteTarget(target));
	} else {
		std::list<StageParams> stage_params_list;		
		StageParams p;
		p.cf_id = OutgoingEdges::CfId();
		p.key = key;
		p.original_key = gn;
		DeleteTarget(target).GetStageParams(p);
		stage_params_list.push_back(p);
		ds_->Put(stage_params_list);
	}
}


// set outgoing edges
void
EdgesInterface::SetOutgoingAllTargets(const graphNodeType &gn,
									 const std::vector<idType> & ids,
									 WriteBatch *wb) {
	if (wb) {
		std::string key;
		buildKey(gn, key);		
		auto outEdges = wb->find(key, OutgoingEdges::CfId());		
		if (!outEdges) {
			outEdges = new OutgoingEdges(); 
			wb->Register(key, outEdges);
		}
		outEdges->Set(ids);
	} else {
		OutgoingEdges outEdges(ids);
		std::list<StageParams> stage_params_list;				
		outEdges.AppendStagesOp(gn, stage_params_list);
		ds_->Put(stage_params_list);
	}
}


// incoming edges 
const std::vector<idType> 
EdgesInterface::GetIncomingEdges(const graphNodeType &gn,
								WriteBatch *wb) const {
	if (wb) {
		std::string key;
		buildKey(gn, key);		
		auto ret = wb->find(key, IncomingEdges::CfId() );
		if (!ret) {
			ret = new IncomingEdges();
			wb->Register(key, ret);
		} 
		if (!ret->fetched()) {
			ret->SetFetchedData(executeGet(ds_, gn, false));
			
		} 
		ret->ApplyChanges();
		return ((IncomingEdges *)ret)->Get();		
	} else {
		return executeGet(ds_, gn, false);
	}		
}
		
	
// add one target
void
EdgesInterface::AddIncomingTarget(const graphNodeType &gn,
								 idType target, 
								 WriteBatch *wb) {
	std::string key;
	buildKey(gn, key);		
	if (wb) {
		auto outEdges = wb->find(key, IncomingEdges::CfId());		
		if (!outEdges) {
			outEdges = new IncomingEdges(); 
			wb->Register(key, outEdges);
		}
		outEdges->addChange(new AddTarget(target));
	} else {
		std::list<StageParams> stage_params_list;		
		StageParams p;
		p.cf_id = IncomingEdges::CfId();
		p.key = key;
		p.original_key = gn;
		AddTarget(target).GetStageParams(p);
		stage_params_list.push_back(p);
		ds_->Put(stage_params_list);
		
	}
	
}
// delete one target
void
EdgesInterface::DeleteIncomingTarget(const graphNodeType &gn,
									idType target,
									WriteBatch *wb) {
	std::string key;
	buildKey(gn, key);		
	if (wb) {
		auto outEdges = wb->find(key, IncomingEdges::CfId());		
		if (!outEdges) {
			outEdges = new IncomingEdges(); 
			wb->Register(key, outEdges);
		}
		outEdges->addChange(new DeleteTarget(target));
	} else {
		std::list<StageParams> stage_params_list;		
		StageParams p;
		p.cf_id = IncomingEdges::CfId();
		p.key = key;
		p.original_key = gn;
		DeleteTarget(target).GetStageParams(p);
		stage_params_list.push_back(p);
		ds_->Put(stage_params_list);
	}
}


// set incoming edges
void
EdgesInterface::SetIncomingAllTargets(const graphNodeType &gn,
									 const std::vector<idType> & ids,
									 WriteBatch *wb) {
	if (wb) {
		std::string key;
		buildKey(gn, key);		
		auto edges = wb->find(key, IncomingEdges::CfId());		
		if (!edges) {
			edges = new IncomingEdges(); 
			wb->Register(key, edges);
		}
		edges->Set(ids);
	} else {
		IncomingEdges edges(ids);
		std::list<StageParams> stage_params_list;				
		edges.AppendStagesOp(gn, stage_params_list);
		ds_->Put(stage_params_list);
	}
}


WriteBatch *
EdgesInterface::newWriteBatch() {
	return new WriteBatch();
}

void
EdgesInterface::CommitWriteBatch(WriteBatch *wb) {
	
	std::list<StageParams> stage_params_list;				
	for (auto obj : *wb) {
		auto gn = extractKey(obj.first.data());
		obj.second->AppendStagesOp(gn , stage_params_list);
		 
		delete obj.second;
	}
	ds_->Put(stage_params_list);
}

void
EdgesInterface::Flush() {
	ds_->Flush();
}

