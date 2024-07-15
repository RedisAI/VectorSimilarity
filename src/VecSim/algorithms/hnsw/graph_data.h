#pragma once
using std::pair;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesList = vecsim_stl::vector<pair<DistType, idType>>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::abstract_priority_queue<DistType, labelType>;


typedef uint16_t linkListSize;
typedef uint8_t elementFlags;
using graphNodeType = pair<idType, ushort>; // represented as: (element_id, level)


// Vectors flags (for marking a specific vector)
typedef enum {
    DELETE_MARK = 0x1, // element is logically deleted, but still exists in the graph
    IN_PROCESS = 0x2,  // element is being inserted into the graph
} Flags;

#pragma pack(1)
struct ElementMetaData {
    labelType label;
    elementFlags flags;

    ElementMetaData(labelType label = SIZE_MAX) noexcept : label(label), flags(IN_PROCESS) {}
};
#pragma pack() // restore default packing

class IncomingEdges : private 	vecsim_stl::vector<idType>  {
public:
	IncomingEdges(std::shared_ptr<VecSimAllocator> allocator) :
		vecsim_stl::vector<idType>(allocator) {}
	void push_back(idType id) {
		vecsim_stl::vector<idType>::push_back(id);
	}
	bool removeIdFromList(idType element_id) {
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
	vecsim_stl::vector<idType> &Get() {
		return *this;
	}
	void save(std::ofstream &output) ;
	void restore(std::ifstream &input);
};
		
	

class LevelData {
public:
    LevelData(std::shared_ptr<VecSimAllocator> allocator)
        : incomingEdges_(allocator),
		  numLinks_(0) {}

	// old interface
    inline void setLinks(vecsim_stl::vector<idType> &links) {
        numLinks_ = links.size();
        memcpy(links_, links.data(), numLinks_ * sizeof(idType));
    }
    template <typename DistType>
    inline void setLinks(candidatesList<DistType> &links) {		
        numLinks_ = 0;
        for (auto &link : links) {
            links_[numLinks_++] = link.second;
        }
    }
	// future interface TBD 
	void AddOutgoingLink(idType);
	void DeleteOutgoingLink(idType);	

    linkListSize numLinks() const {return numLinks_;}
    linkListSize &numLinks() {return numLinks_;}
	idType link(linkListSize i) const {
		assert (i < numLinks_);
		return links_[i];
	}
	idType &link(linkListSize i)  {
		assert (i < numLinks_);
		return links_[i];
	}
	
	void save(std::ofstream &output) ;
	void restore(std::ifstream &input);
	IncomingEdges *incomingEdges() {return &incomingEdges_;}
private:
	
    IncomingEdges incomingEdges_;
    linkListSize numLinks_;
    // Flexible array member - https://en.wikipedia.org/wiki/Flexible_array_member
    // Using this trick, we can have the links list as part of the LevelData struct, and avoid
    // the need to dereference a pointer to get to the links list.
    // We have to calculate the size of the struct manually, as `sizeof(LevelData)` will not include
    // this member. We do so in the constructor of the index, under the name `levelDataSize` (and
    // `elementGraphDataSize`). Notice that this member must be the last member of the struct and
    // all nesting structs.
    idType links_[];

};

struct ElementGraphData {
    size_t toplevel;
    std::mutex neighborsGuard;
    LevelData *others;
    LevelData level0;

    ElementGraphData(size_t maxLevel, size_t high_level_size,
                     std::shared_ptr<VecSimAllocator> allocator)
        : toplevel(maxLevel), others(nullptr), level0(allocator) {
        if (toplevel > 0) {
            others = (LevelData *)allocator->callocate(high_level_size * toplevel);
            if (others == nullptr) {
                throw std::runtime_error("VecSim index low memory error");
            }
            for (size_t i = 0; i < maxLevel; i++) {
                new ((char *)others + i * high_level_size) LevelData(allocator);
            }
        }
    }
    ~ElementGraphData() = delete; // Should be destroyed using `destroyGraphData`
	void lock() { neighborsGuard.lock();}
	void unlock() {neighborsGuard.unlock();}
	void restoreLevels(std::ifstream &input);
	
	
};

struct IndexMetaData
{
    size_t M;
    size_t M0;
    size_t efConstruction;

    // Index search parameter
	// TBD (Alon) are those realy part of the graph or should be part of the job def. 
    size_t ef;
    double epsilon;

    // Index meta-data (based on the data dimensionality and index parameters)
    size_t elementGraphDataSize;
    size_t levelDataSize;
	size_t vectorDataSize;
    double mult;
    // Index global state - these should be guarded by the indexDataGuard lock in
    // multithreaded scenario.
    size_t curElementCount;
	size_t numMarkedDeleted_;
    idType entrypointNode;
    size_t maxLevel; // this is the top level of the entry point's element
	size_t blockSize_;
	size_t alignment_;	   
	size_t maxElements;
	void restore(std::ifstream &input);
	void save(std::ofstream &output) const;
};


class GraphData  {
public:
	GraphData(std::shared_ptr<VecSimAllocator> allocator, IndexMetaData &indexMetaData) :
		vectorBlocks_(allocator),
		graphDataBlocks_(allocator),
		idToMetaData_(allocator),
		allocator_(allocator),
		indexMetaData(indexMetaData)
		{			
		}
	void Init() {
		idToMetaData_.resize(indexMetaData.maxElements);
		auto initial_vector_size = indexMetaData.maxElements / blockSize();
			
		
		vectorBlocks_.reserve(initial_vector_size);
		graphDataBlocks_.reserve(initial_vector_size);
	}

	size_t blockSize() const { return indexMetaData.blockSize_;}
		
	const char *getDataByInternalId(idType internal_id) const {
		return vectorBlocks_[internal_id / blockSize()].getElement(internal_id % blockSize());
	}
	
	
	ElementGraphData *getGraphDataByInternalId(idType internal_id) const {
		return (ElementGraphData *)graphDataBlocks_[internal_id / blockSize()].getElement(
			internal_id % blockSize());
	}

	void pushVector(const void *vector_data, ElementGraphData *cur_egd) {
		
		vectorBlocks_.back().addElement(vector_data);
		graphDataBlocks_.back().addElement(cur_egd);
	}

	
	LevelData &getLevelData(idType internal_id, size_t level) const {
		return getLevelData(getGraphDataByInternalId(internal_id), level);
	}

	LevelData &getLevelData(idType internal_id, size_t level)  {
		return getLevelData(getGraphDataByInternalId(internal_id), level);
	}



	LevelData &getLevelData(ElementGraphData *elem, size_t level) const {
		assert(level <= elem->toplevel);
		if (level == 0) {
			return elem->level0;
		} else {
			return *(LevelData *)((char *)elem->others + (level - 1) * levelDataSize());
		}
	}
	

    labelType label(idType internal_id) const {
        return idToMetaData_[internal_id].label;
    }
	labelType label(idType internal_id) {
        return idToMetaData_[internal_id].label;
    }
	void addLabel(idType internal_id, labelType l) {
		if (idToMetaData_.size() > internal_id) {
			idToMetaData_[internal_id] = l;
		} else {
			assert(idToMetaData_.size() == internal_id);
			idToMetaData_.push_back(l);
		}
	}
	size_t numMetaData() {
		return idToMetaData_.size();
	}
	void copyMetaData(idType from, idType to) {
		idToMetaData_[to] = idToMetaData_[from];
	}
	
    // Flagging API
    template <Flags FLAG>
    inline void markAs(idType internalId) {
        __atomic_fetch_or(&idToMetaData_[internalId].flags, FLAG, 0);
    }
    template <Flags FLAG>
    inline void unmarkAs(idType internalId) {
        __atomic_fetch_and(&idToMetaData_[internalId].flags, ~FLAG, 0);
    }
    template <Flags FLAG>
    inline bool isMarkedAs(idType internalId) const {
        return idToMetaData_[internalId].flags & FLAG;
    }
	bool isMarkedDeleted(idType internalId) const {
		return isMarkedAs<DELETE_MARK>(internalId);
	}


	bool isInProcess(idType internalId) const {
		return isMarkedAs<IN_PROCESS>(internalId);
	}

	void unmarkInProcess(idType internalId) {
		// Atomically unset the IN_PROCESS mark flag (note that other parallel threads may set the flags
		// at the same time (for marking the element with MARK_DELETE flag).
		unmarkAs<IN_PROCESS>(internalId);
	}

	
    void multiGet(const LevelData &) const;
    void multiGet(const std::vector<idType> &) const;

	void resizeIndexCommon(size_t new_max_elements){
		idToMetaData_.resize(new_max_elements);
		idToMetaData_.shrink_to_fit();
	}
	void growByBlock();


	void shrinkByBlock();
		
	void shrinkToFit() {
      vectorBlocks_.shrink_to_fit();
	  graphDataBlocks_.shrink_to_fit();
	  idToMetaData_.shrink_to_fit();
	}

	
	size_t levelDataSize() const {return indexMetaData.levelDataSize;}
	
	const ElementMetaData *getMetaDataAddress(idType internal_id) {
		return idToMetaData_.data() + internal_id;
	}
	void fitMemory() {
        idToMetaData_.shrink_to_fit();		
	}
	const vecsim_stl::vector<ElementMetaData> &idToMetaData() const {
		return 	idToMetaData_;
	}
	idType GetNewEntryPoint(idType &candidates) const;
	void replaceEntryPoint();
	
	void getLastElement(char *&last_element_data,ElementGraphData *&last_element) {
		DataBlock &last_vector_block = vectorBlocks_.back();
		last_element_data = last_vector_block.removeAndFetchLastElement();
		DataBlock &last_gd_block = graphDataBlocks_.back();
		last_element = (ElementGraphData *)last_gd_block.removeAndFetchLastElement();
	}

public:
	
	void save(std::ofstream &output) const;
	void restore(std::ifstream &input);
	
	
	
private:
	vecsim_stl::vector<DataBlock> vectorBlocks_;
	vecsim_stl::vector<DataBlock> graphDataBlocks_;
	vecsim_stl::vector<ElementMetaData> idToMetaData_;
	std::shared_ptr<VecSimAllocator> allocator_;
	IndexMetaData &indexMetaData;
};
