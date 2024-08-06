#pragma once
#include "write_batch.h"


class AddTarget : public ObjectChange
{
public:
	AddTarget(idType target) : target_(target) {};
	void GetStageParams(StageParams &stage)	 override {
		stage.op = StageParams::UPDATE_DATA;
		stage.data = std::string("A") + std::string((char *)&target_, sizeof(idType)); 
	}
	void Apply(std::vector<idType> &data) override {
		data.push_back(target_);
	}
	void Apply(size_t &size, idType *cur_value) override {
		cur_value[size] = target_;		
		size++;
	}
		
private:
	idType target_;	
};

class DeleteTarget : public ObjectChange
{
public:
	DeleteTarget(idType target) : target_(target) {};
	void GetStageParams(StageParams &stage)	{
		stage.op = StageParams::UPDATE_DATA;
		stage.data = std::string("D") + std::string((char *)&target_, sizeof(idType));
	}
	void Apply(std::vector<idType> &data) override {
		auto iter = std::find(data.begin(), data.end(), target_);
		if (iter != data.end()) {
			*iter = data.back();
			data.resize(data.size() -1);
		}
	}	
	void Apply(size_t &size, idType *cur_value) override {
		for (size_t i = 0; i < size; i++) {
			if (cur_value[i] == target_) {
				size--;
				cur_value[i] = cur_value[size];
				return;
			}
		}
	}
private:
	idType target_;	
};




class Edges : public DirtyObject
{
public:
	virtual ~Edges() {}

	Edges(const std::vector<idType>  &from) : DirtyObject(from) {
	}
	Edges() : DirtyObject() {
	}

	void
	AppendStagesOp(const graphNodeType &gn,
				   std::list<StageParams> &stageParmsList) override {
		if (was_fetched_) {
			ApplyChanges();
		}
		if (is_dirty_) {			
			StageParams p;
			p.cf_id = GetCfId();
			p.op =  StageParams::WRITE_DATA;
			p.original_key = gn;
			buildKey(gn, p.key);
			p.data = std::string((const char *) object_value_.data(),
								 object_value_.size() * sizeof(idType));
			stageParmsList.push_back(p);
			
			
		} else {
			for (auto ch: all_changes_) {				
				StageParams p;
				p.cf_id = GetCfId();
				p.original_key = gn;
				buildKey(gn, p.key);
				ch->GetStageParams(p);
				stageParmsList.push_back(p);
				delete ch;
			}
			all_changes_.clear();   
		}
	}

	
};
	



class OutgoingEdges : public Edges
{
public:
	~OutgoingEdges() {};
	OutgoingEdges(const std::vector<idType>  &from) : Edges(from) {
	}
	OutgoingEdges()  {
	}
	virtual CF_ID GetCfId() override {return CfId();};
	static CF_ID CfId() {return OUTGOING_CF;}

	static void
	PrepareGetsParams(const graphNodeType &gn,
					  std::list<GetParams> &getParmsList)  {
		
		GetParams p;
		p.cf_id = CfId();
		p.original_key = gn;
		buildKey(gn, p.key);
		getParmsList.push_back(p);
	}
};

class IncomingEdges : public Edges
{
public:
	~IncomingEdges(){};
	IncomingEdges(const std::vector<idType>  &from) : Edges(from) {
	}
	IncomingEdges() : Edges() {
	}
	
	virtual CF_ID GetCfId() override  {return CfId();};
	static CF_ID CfId() {return INCOMING_CF;}

	static void
	PrepareGetsParams(const graphNodeType &gn,
					  std::list<GetParams> &getParmsList)  {
		
		GetParams p;
		p.cf_id = CfId();
		p.original_key = gn;
		buildKey(gn, p.key);
		getParmsList.push_back(p);
	}
};

