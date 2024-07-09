#include "hnsw.h"
#include "VecSim/utils/serializer.h"

void GraphData::growByBlock() {
    // Validations
    assert(vectorBlocks_.size() == graphDataBlocks_.size());
    assert(vectorBlocks_.size() == 0 || vectorBlocks_.back().getLength() == blockSize());

    vectorBlocks_.emplace_back(blockSize(), indexMetaData.vectorDataSize, allocator_, indexMetaData.alignment_);
    graphDataBlocks_.emplace_back(blockSize(), indexMetaData.elementGraphDataSize, allocator_);

}

void GraphData::shrinkByBlock() {
    assert(indexMetaData.maxElements >= this->blockSize());
    // Validations
    assert(vectorBlocks_.size() == graphDataBlocks_.size());
    assert(vectorBlocks_.size() > 0);
    assert(vectorBlocks_.back().getLength() == 0);

    vectorBlocks_.pop_back();
    graphDataBlocks_.pop_back();
}
#ifdef BUILD_TESTS
void GraphData::save(std::ofstream &output) const {
	indexMetaData.save(output);
	
	
}
void GraphData::restore(std::ifstream &input) {
	// indexMetaData.restore(input); called during construct TBD
	

    // Restore id to metadata vector
    labelType label = 0;
    elementFlags flags = 0;
    for (idType id = 0; id < indexMetaData.curElementCount; id++) {
        Serializer::readBinaryPOD(input, label);
        Serializer::readBinaryPOD(input, flags);
        idToMetaData_[id].label = label;
        idToMetaData_[id].flags = flags;		
    }
    // Get number of blocks
    unsigned int num_blocks = 0;
    Serializer::readBinaryPOD(input, num_blocks);
    this->vectorBlocks_.reserve(num_blocks);
    this->graphDataBlocks_.reserve(num_blocks);

    // Get data blocks
    for (size_t i = 0; i < num_blocks; i++) {
		auto dataSize = indexMetaData.vectorDataSize;
        vectorBlocks_.emplace_back(blockSize(),
								   dataSize,
								   allocator_,
								   indexMetaData.alignment_);
        unsigned int block_len = 0;
        Serializer::readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            char cur_vec[dataSize];
            input.read(cur_vec, dataSize);
            this->vectorBlocks_.back().addElement(cur_vec);
        }
    }

    // Get graph data blocks
    ElementGraphData *cur_egt;
	auto graphDataSize = indexMetaData.elementGraphDataSize;
    char tmpData[graphDataSize];
    size_t toplevel = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        this->graphDataBlocks_.emplace_back(blockSize(),
											graphDataSize,
											allocator_);
        unsigned int block_len = 0;
        Serializer::readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            // Reset tmpData
            memset(tmpData, 0, graphDataSize);
            // Read the current element top level
            Serializer::readBinaryPOD(input, toplevel);
            // Allocate space and structs for the current element
            try {
                new (tmpData) ElementGraphData(toplevel,
											   indexMetaData.levelDataSize,
											   allocator_);
            } catch (std::runtime_error &e) {
				
                printf(VecSimCommonStrings::LOG_WARNING_STRING,
                          "Error - allocating memory for new element failed due to low memory");
                throw e;
            }
            // Add the current element to the current block, and update cur_egt to point to it.
            graphDataBlocks_.back().addElement(tmpData);
            cur_egt = (ElementGraphData *)graphDataBlocks_.back().getElement(j);

            // Restore the current element's graph data
			for (size_t i = 0 ; i <= toplevel; i++)
				getLevelData(cur_egt, i).restore(input);			
        }
    }

}

void LevelData::restore(std::ifstream &input) {		
	Serializer::readBinaryPOD(input, numLinks_);
	for (size_t i = 0; i < numLinks_; i++) {
		Serializer::readBinaryPOD(input, links_[i]);
	}
	
	// Restore the incoming edges of the current element
	unsigned int size;
	Serializer::readBinaryPOD(input, size);
	incomingEdges_->reserve(size);
	idType id = INVALID_ID;
	for (size_t i = 0; i < size; i++) {
		Serializer::readBinaryPOD(input, id);
		incomingEdges_->push_back(id);
    }
}

#endif

void GraphData::replaceEntryPoint()
{
	idType old_entry_point_id = indexMetaData.entrypointNode;
    auto *old_entry_point = getGraphDataByInternalId(old_entry_point_id);

    // Sets an (arbitrary) new entry point, after deleting the current entry point.
    while (old_entry_point_id == indexMetaData.entrypointNode) {
        // Use volatile for this variable, so that in case we would have to busy wait for this
        // element to finish its indexing, the compiler will not use optimizations. Otherwise,
        // the compiler might evaluate 'isInProcess(candidate_in_process)' once instead of calling
        // it multiple times in a busy wait manner, and we'll run into an infinite loop if the
        // candidate is in process when we reach the loop.
        volatile idType candidate_in_process = INVALID_ID;

        // Go over the entry point's neighbors at the top level.
        old_entry_point->lock();
        LevelData &old_ep_level = getLevelData(old_entry_point, indexMetaData.maxLevel);
        // Tries to set the (arbitrary) first neighbor as the entry point which is not deleted,
        // if exists.
        for (size_t i = 0; i < old_ep_level.numLinks(); i++) {
            if (!isMarkedDeleted(old_ep_level.link(i))) {
                if (!isInProcess(old_ep_level.link(i))) {
                    indexMetaData.entrypointNode = old_ep_level.link(i);
                    old_entry_point->unlock();
                    return;
                } else {
                    // Store this candidate which is currently being inserted into the graph in
                    // case we won't find other candidate at the top level.
                    candidate_in_process = old_ep_level.link(i);
                }
            }
        }
        old_entry_point->unlock();

        // If there is no neighbors in the current level, check for any vector at
        // this level to be the new entry point.
        idType cur_id = 0;
        for (DataBlock &graph_data_block : graphDataBlocks_) {
            size_t size = graph_data_block.getLength();
            for (size_t i = 0; i < size; i++) {
                auto cur_element = (ElementGraphData *)graph_data_block.getElement(i);
                if (cur_element->toplevel == indexMetaData.maxLevel &&
					cur_id != old_entry_point_id &&
                    !isMarkedDeleted(cur_id)) {
                    // Found a non element in the current max level.
                    if (!isInProcess(cur_id)) {
                        indexMetaData.entrypointNode = cur_id;
                        return;
                    } else if (candidate_in_process == INVALID_ID) {
                        // This element is still in process, and there hasn't been another candidate
                        // in process that has found in this level.
                        candidate_in_process = cur_id;
                    }
                }
                cur_id++;
            }
        }
        // If we only found candidates which are in process at this level, do busy wait until they
        // are done being processed (this should happen in very rare cases...). Since
        // candidate_in_process was declared volatile, we can be sure that isInProcess is called in
        // every iteration.
        if (candidate_in_process != INVALID_ID) {
            while (isInProcess(candidate_in_process))
                ;
            indexMetaData.entrypointNode = candidate_in_process;
            return;
        }
        // If we didn't find any vector at the top level, decrease the maxLevel and try again,
        // until we find a new entry point, or the index is empty.
        assert(old_entry_point_id == indexMetaData.entrypointNode);
        indexMetaData.maxLevel--;
        if ((int)indexMetaData.maxLevel < 0) {
            indexMetaData.maxLevel = HNSW_INVALID_LEVEL;
            indexMetaData.entrypointNode = INVALID_ID;
        }
    }


}

void GraphData::multiGet(const LevelData &levelData) const {
	//never fetch more than 512K due to limit L2 size(should be enough)
	size_t num_fetched = std::min<size_t>(levelData.numLinks(),
										  512*1024/indexMetaData.levelDataSize);
	
	for (size_t i = 0; i < num_fetched; i++) {
		auto addr = getDataByInternalId(levelData.link(i));
		for (size_t offset = 0; offset < indexMetaData.levelDataSize; offset += 32) {
			__builtin_prefetch(addr + offset);
		}
	}
}

	

void IndexMetaData::restore(std::ifstream &input) {
    // Restore index build parameters
    Serializer::readBinaryPOD(input, this->M);
    Serializer::readBinaryPOD(input, this->M0);
    Serializer::readBinaryPOD(input, this->efConstruction);

    // Restore index search parameter
    Serializer::readBinaryPOD(input, this->ef);
    Serializer::readBinaryPOD(input, this->epsilon);

    // Restore index meta-data
    this->elementGraphDataSize = sizeof(ElementGraphData) + sizeof(idType) * this->M0;
    this->levelDataSize = sizeof(LevelData) + sizeof(idType) * this->M;
    Serializer::readBinaryPOD(input, this->mult);

    // Restore index state
    Serializer::readBinaryPOD(input, this->curElementCount);
    Serializer::readBinaryPOD(input, this->numMarkedDeleted_);
    Serializer::readBinaryPOD(input, this->maxLevel);
    Serializer::readBinaryPOD(input, this->entrypointNode);
}

void IndexMetaData::save(std::ofstream &output) const {
	
    // Save index type
    Serializer::writeBinaryPOD(output, VecSimAlgo_HNSWLIB);
#if 0
    // Save VecSimIndex fields
    Serializer::writeBinaryPOD(output, this->dim_);
    Serializer::writeBinaryPOD(output, this->vecType);
    Serializer::writeBinaryPOD(output, this->metric);
    Serializer::writeBinaryPOD(output, this->blockSize_);
    Serializer::writeBinaryPOD(output, this->isMulti);
    Serializer::writeBinaryPOD(output, this->maxElements); // This will be used to restore the index initial
                                               // capacity

    // Save index build parameters
    Serializer::writeBinaryPOD(output, this->M);
    Serializer::writeBinaryPOD(output, this->M0);
    Serializer::writeBinaryPOD(output, this->efConstruction);

    // Save index search parameter
    Serializer::writeBinaryPOD(output, this->ef);
    Serializer::writeBinaryPOD(output, this->epsilon);

    // Save index meta-data
    Serializer::writeBinaryPOD(output, this->mult);

    // Save index state
    Serializer::writeBinaryPOD(output, this->curElementCount);
    Serializer::writeBinaryPOD(output, this->numMarkedDeleted);
    Serializer::writeBinaryPOD(output, this->maxLevel);
    Serializer::writeBinaryPOD(output, this->entrypointNode);
#endif
}


void LevelData::save(std::ofstream &output)  {
    // Save the links of the current element
    Serializer::writeBinaryPOD(output, numLinks());
    for (size_t i = 0; i < numLinks(); i++) {
        Serializer::writeBinaryPOD(output, link(i));
    }

    // Save the incoming edges of the current element
    unsigned int size = incomingEdges()->size();
    Serializer::writeBinaryPOD(output, size);
    for (idType id : *incomingEdges()) {
        Serializer::writeBinaryPOD(output, id);
    }

    // Shrink the incoming edges vector for integrity check
    incomingEdges()->shrink_to_fit();
}
