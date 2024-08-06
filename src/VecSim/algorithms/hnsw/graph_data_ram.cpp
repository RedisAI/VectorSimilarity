#include "graph_data_ram.h"
struct LevelDataOnRam {
    static size_t max_num_outgoing_links;
    LevelDataOnRam(std::shared_ptr<VecSimAllocator> allocator) :
        incomingEdges(allocator), outgoingEdges() {
    }
    static size_t
    GetAllocationSizeBytes() {
        return sizeof(incomingEdges) + sizeof(outgoingEdges) +
            max_num_outgoing_links * sizeof(idType);
    }
    // currently only one size of level data
    IncomingEdges incomingEdges;
    OutgoingEdges outgoingEdges; // must be last
    
    
};
size_t LevelDataOnRam::max_num_outgoing_links;

struct VectorGraphData {
    VectorGraphData(std::shared_ptr<VecSimAllocator> allocator,
                     size_t num_levels) :
        level0_data(allocator) {
        if (num_levels == 0) {
            others = nullptr;
        } else {
            others = (char *)allocator->callocate(
                LevelDataOnRam::GetAllocationSizeBytes() * num_levels);
        }
    }
        
        
    LevelDataOnRam &getLevelData(size_t level_num) {
        if (level_num == 0) return level0_data;
        // else
        return *(LevelDataOnRam *)
            (others + (level_num-1) * LevelDataOnRam::GetAllocationSizeBytes());
    }
    static size_t
    GetAllocationSizeBytes() {
        return sizeof(char *) + LevelDataOnRam::GetAllocationSizeBytes();
    };
                                    
    char *others;
    // since level0_data has a variable size it must be last 
    LevelDataOnRam level0_data;
};


 RamGraphData::RamGraphData(std::shared_ptr<VecSimAllocator> allocator,
                            size_t block_size,
                            size_t max_num_outgoing_links,
                            size_t vector_size_bytes,
                            size_t initial_capacity,
                            size_t vector_alignment) :
     vectorBlocks_(allocator),
     graphDataBlocks_(allocator),
     idToMetaData_(allocator),
     allocator_(allocator),
     block_size_(block_size),
     vector_size_bytes_(vector_size_bytes),
     vector_alignment_(vector_alignment)
 {
     LevelDataOnRam::max_num_outgoing_links = max_num_outgoing_links;
     if (initial_capacity)  {
         idToMetaData_.reserve(initial_capacity);
         auto initial_vector_size = initial_capacity / block_size_;
         vectorBlocks_.reserve(initial_vector_size);
         graphDataBlocks_.reserve(initial_vector_size);
     }
 }


const char *
RamGraphData::getVectorByInternalId(idType internal_id) const {
    return vectorBlocks_[internal_id / block_size_].getElement(internal_id % block_size_);
}
    
void
RamGraphData::multiGetVectors(const std::vector<idType> &ids,
				std::vector<const char *> &results) const {
	results.reserve(ids.size());
	for (auto id:ids) {
		results.push_back(getVectorByInternalId(id));
	}
}
	
    
         
idType
RamGraphData::pushVector(const void *vector_data,
                         int max_level,
                         const labelType  &label,
                         WriteBatch *wb) {
    idToMetaData_.push_back(VectorMetaData(label,max_level));
    
    if (vectorBlocks_.size() == 0 ||
        vectorBlocks_.back().getLength() == block_size_) {
        growByBlock();
    
    }            
    idType ret = vectorBlocks_.size() * block_size_ +
        vectorBlocks_.back().getLength();
    assert(idToMetaData_.size() == ret);

    vectorBlocks_.back().addElement(vector_data);

    VectorGraphData tmp(allocator_, max_level);
    graphDataBlocks_.back().addElement(&tmp);
    
    return ret;
}

// outgoing edges 
const absEdges &
RamGraphData::GetLevelOutgoingEdges(const graphNodeType &gn) const {
    return getGraphDataByInternalId(gn.first)->
        getLevelData(gn.second).outgoingEdges;
}

absEdges &
RamGraphData::GetLevelOutgoingEdges(const graphNodeType &gn,
    WriteBatch *)  {
    return getGraphDataByInternalId(gn.first)->
        getLevelData(gn.second).outgoingEdges;
}
// incoming edges 
const absEdges &
RamGraphData::GetLevelIncomingEdges(const graphNodeType &gn) const {
    return getGraphDataByInternalId(gn.first)->
        getLevelData(gn.second).incomingEdges;
}

absEdges &
RamGraphData::GetLevelIncomingEdges(const graphNodeType &gn,
    WriteBatch *)  {
    return getGraphDataByInternalId(gn.first)->
        getLevelData(gn.second).incomingEdges;
}

idType
RamGraphData::getVectorIdByLevel(short level,
                                 idType startingId) const {
    for (idType i = startingId; i < idToMetaData_.size(); i++) {
        auto const &v =  vectorMetaDataById(i);
        if (v.max_level_ == level) {
            return i;
        }
    }
    for (idType i = 0; i < startingId; i++) {
        auto const &v =  vectorMetaDataById(i);
        if (v.max_level_ == level) {
            return i;
        }
    }
    return idType(-1);
}

idType 
RamGraphData::getGarbadgeCollectionTarget(idType startingId) const {
    for (idType i = startingId; i < idToMetaData_.size(); i++) {
        auto const &v =  vectorMetaDataById(i);
        if (v.ismarked(VectorMetaData::PERMANENT_DELETED)) {
            return i;
        }
    }
    return idType(-1);
}


VectorGraphData *
RamGraphData::getGraphDataByInternalId(idType internal_id) const {
    return (VectorGraphData *)
        graphDataBlocks_[internal_id / block_size_].
        getElement(internal_id % block_size_);
}

void RamGraphData::growByBlock() {
    // Validations
    assert(vectorBlocks_.size() == graphDataBlocks_.size());
    assert(vectorBlocks_.size() == 0 ||
           vectorBlocks_.back().getLength() == block_size_);

    vectorBlocks_.emplace_back(block_size_, vector_size_bytes_,
                               allocator_, vector_alignment_);
    graphDataBlocks_.emplace_back(block_size_,
                                  VectorGraphData::GetAllocationSizeBytes(),
                                  allocator_);
}

void RamGraphData::shrinkByBlock() {
    assert(vectorBlocks_.size() == graphDataBlocks_.size());
    assert(vectorBlocks_.size() > 0);
    assert(vectorBlocks_.back().getLength() == 0);
    
    vectorBlocks_.pop_back();
    graphDataBlocks_.pop_back();
}

void RamGraphData::shrinkToFit() {
    while (vectorBlocks_.size() && vectorBlocks_.back().getLength() == 0) {
        shrinkByBlock();
    }
    vectorBlocks_.shrink_to_fit();
    graphDataBlocks_.shrink_to_fit();
    idToMetaData_.shrink_to_fit();
}

    
void RamGraphData::save(std::ofstream &) const {
}
    
void RamGraphGrestore(std::ifstream &) {
    // TBD
}
    
