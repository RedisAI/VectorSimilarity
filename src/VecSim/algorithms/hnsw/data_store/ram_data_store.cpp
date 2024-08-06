#include "abs_data_store.h"
#include "write_batch.h"
#include <mutex>
class RamIncomingEdges : private  std::vector<idType> {
public:
    RamIncomingEdges(std::shared_ptr<VecSimAllocator> ) :
        std::vector<idType>() {}

	void ApplyChange(ObjectChange &ch) {
		ch.Apply(*this);
	}
	void Set(const std::string &ids) {
		resize(ids.size() / sizeof(idType));
		std::memcpy(data(), ids.data(), ids.size() * sizeof(idType));
	}
	const std::vector<idType> &Get() const {return *this;}

    void save(std::ofstream &output);
    void restore(std::ifstream &input);
};


class RamOutgoingEdges  {
public:
	static size_t max_num_outgoing_links;
	static size_t GetAllocationSize() {
		return sizeof(num_links_) + sizeof(idType) *max_num_outgoing_links;
	}
			
    RamOutgoingEdges() :num_links_(0) {}
    ~RamOutgoingEdges() = default;
    
    void
    Set(const std::string &ids)  {
        num_links_ = ids.size() / sizeof(idType);
		std::memcpy(links_, ids.data(), ids.size() * sizeof(idType));
    }

	const std::vector<idType> Get() const {
		std::vector<idType> ret(num_links_);
		std::memcpy(ret.data(), links_, num_links_ * sizeof(idType));
		return ret;
	}
	void ApplyChange(ObjectChange &ch) {
		ch.Apply(num_links_, links_);
	}
	

    void save(std::ofstream &output);
    void restore(std::ifstream &input);
private:
    size_t num_links_=0;
    idType links_[];
};

size_t RamOutgoingEdges::max_num_outgoing_links;


class RamLevelData {
public:
	RamLevelData(std::shared_ptr<VecSimAllocator> allocator) : incoming_edges(allocator) {
	}
	static size_t GetAllocationSize() {
		return sizeof(RamLevelData) + RamOutgoingEdges::GetAllocationSize();
	}
									 
	RamIncomingEdges incoming_edges;
	RamOutgoingEdges outgoing_edges;
};
	
	

class RamDataStore : public EdgesDataStore {
public:
	RamDataStore(std::shared_ptr<VecSimAllocator> allocator,
				 size_t block_size,
				 size_t max_num_outgoing_links,
				 size_t initial_capacity) :
		graphDataBlocks_(allocator),
		allocator_(allocator),
		block_size_(block_size)		{

		RamOutgoingEdges::max_num_outgoing_links = max_num_outgoing_links;
		if (initial_capacity)  {
			auto initial_vector_size = initial_capacity / block_size_;
			graphDataBlocks_.reserve(initial_vector_size);
		}
	}
	void Get(std::list<GetParams> &get_params_list) override;
	void Put(const std::list<StageParams> &stage_params_list) override;


	
private:
	RamLevelData *GetLevel0Data(const idType id) {
		return (RamLevelData *)graphDataBlocks_[id / block_size_].
			getElement(id % block_size_);
	}
	size_t getCapacity() const {
		return graphDataBlocks_.size() * block_size_;
	}
	void growByBlock() {
		graphDataBlocks_.emplace_back(block_size_,
									  RamLevelData::GetAllocationSize(),
									  allocator_);
	}
	void HandleDelete(idType id) {
		if (id == getCapacity() -1) {
			// TBD shrink the graph data blocks?
		}
		for (size_t index = 0; index <  other_levels_map.size(); index++) {
			auto iter = other_levels_map[index]->find(id);
			if (iter != other_levels_map[index]->end()) {
				// TBD ALON ... allocator_.free(iter->second);
				other_levels_map[index]->erase(iter);				
			} else {
				// TBD 
				return; 
			}
		}
			
	}


				
private:
    vecsim_stl::vector<DataBlock> graphDataBlocks_;
    std::shared_ptr<VecSimAllocator> allocator_;    
    const size_t block_size_;
	using edges_map = std::unordered_map<idType, RamLevelData *>;
	std::vector< edges_map *> other_levels_map; 
	std::shared_mutex mutex; 
};
	

void RamDataStore::Put(const std::list<StageParams> &stage_params_list) 
 {
	 std::unique_lock single_writer(mutex);
	 for (auto stage_param : stage_params_list) {
		 auto gn = stage_param.original_key;
		 auto level = gn.second;
		 auto id = gn.first;
		 if (stage_param.op == StageParams::DELETE_DATA) {
			 HandleDelete(id);
		 } else {
			 RamLevelData *lv = nullptr;
			 if  (level == 0) {			 
				 while (id >= getCapacity())
					 growByBlock();
				 lv = GetLevel0Data(id);
			 } else  {
				 auto index = level -1;
				 if (level > other_levels_map.size()) {
					 size_t cur_size = other_levels_map.size();
					 other_levels_map.resize(level);
					 for (; cur_size  < level; cur_size++) {
						 other_levels_map[index] = new std::unordered_map<idType, RamLevelData *> ;
					 }
				 }
				 auto iter = other_levels_map[index]->find(id);
				 if (iter == other_levels_map[index]->end()) {
					 //allocate
					 auto space =  (char *)allocator_->callocate(
						 RamLevelData::GetAllocationSize());
					 lv = new (space) RamLevelData(allocator_);
					 other_levels_map[level]->insert({id, lv});
				 } else {
					 lv = iter->second;
				 }
			 }
		 
			 if (stage_param.op == StageParams::WRITE_DATA) {
				 if (stage_param.cf_id == OUTGOING_CF) {
					 lv->outgoing_edges.Set(stage_param.data);
				 } else { 
					 lv->incoming_edges.Set(stage_param.data);
				 } 
			 } else if (stage_param.op == StageParams::UPDATE_DATA) {
				 auto ch = ObjectChange::NewObjectChangeFromString(stage_param.data.data());
				 if (stage_param.cf_id == OUTGOING_CF) {
					 lv->outgoing_edges.ApplyChange(*ch);
				 } else { 
					 lv->incoming_edges.ApplyChange(*ch);
				 }
				 delete ch;
			 }
		 }
	 }
 }

void RamDataStore::Get(std::list<GetParams> &get_params_list) {
	 std::shared_lock multi_readers(mutex);
	 for (auto &get_param : get_params_list) {
		 auto gn = get_param.original_key;
		 auto level = gn.second;
		 auto id = gn.first;
		 RamLevelData *lv = nullptr;
		 if (level == 0) {
			 if (id < getCapacity() ) {
				 lv = GetLevel0Data(id);
			 }
		 } else {
			 size_t index = level -1;
			 if (index < other_levels_map.size()) {
				 auto iter = other_levels_map[index]->find(id);
				 if (iter != other_levels_map[index]->end()) {
					 lv = iter->second;
				 }
			 }
		 }
		 if (lv) {
			 if (get_param.cf_id == OUTGOING_CF) {
				 get_param.data = lv->outgoing_edges.Get();
			 } else { 
				 get_param.data = lv->incoming_edges.Get();
			 }
		 }
	 }
			 
}

EdgesDataStore *NewRamDataStore(std::shared_ptr<VecSimAllocator> allocator,
								size_t block_size,
								size_t max_num_outgoing_links,
								size_t initial_capacity) {
	return new RamDataStore(allocator, block_size, max_num_outgoing_links, initial_capacity);
}

			 
		 

	 


