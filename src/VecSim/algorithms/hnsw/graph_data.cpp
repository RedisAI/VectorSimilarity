#include "hnsw.h"
#include "VecSim/utils/serializer.h"

void GraphData::growByBlock() {
    // Validations
    assert(vectorBlocks_.size() == graphDataBlocks_.size());
    assert(vectorBlocks_.size() == 0 || vectorBlocks_.back().getLength() == blockSize());

    vectorBlocks_.emplace_back(blockSize(), indexMetaData.levelDataSize, allocator_, indexMetaData.alignment_);
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
void GraphData::save(std::ofstream &) const {
}
void GraphData::restore(std::ifstream &) {
}

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

	

#if BUILD_TEST
void IndexMetaData::restore(std::ifstream &input) {
    // Restore index build parameters
    readBinaryPOD(input, this->M);
    readBinaryPOD(input, this->M0);
    readBinaryPOD(input, this->efConstruction);

    // Restore index search parameter
    readBinaryPOD(input, this->ef);
    readBinaryPOD(input, this->epsilon);

    // Restore index meta-data
    this->elementGraphDataSize = sizeof(ElementGraphData) + sizeof(idType) * this->M0;
    this->levelDataSize = sizeof(LevelData) + sizeof(idType) * this->M;
    readBinaryPOD(input, this->mult);

    // Restore index state
    readBinaryPOD(input, this->curElementCount);
    readBinaryPOD(input, this->numMarkedDeleted);
    readBinaryPOD(input, this->maxLevel);
    readBinaryPOD(input, this->entrypointNode);
}

void IndexMetaData::save(std::ofstream &output) const {
    // Save index type
    writeBinaryPOD(output, VecSimAlgo_HNSWLIB);
    // Save VecSimIndex fields
    writeBinaryPOD(output, this->dim);
    writeBinaryPOD(output, this->vecType);
    writeBinaryPOD(output, this->metric);
    writeBinaryPOD(output, this->blockSize);
    writeBinaryPOD(output, this->isMulti);
    writeBinaryPOD(output, this->maxElements); // This will be used to restore the index initial
                                               // capacity

    // Save index build parameters
    writeBinaryPOD(output, this->M);
    writeBinaryPOD(output, this->M0);
    writeBinaryPOD(output, this->efConstruction);

    // Save index search parameter
    writeBinaryPOD(output, this->ef);
    writeBinaryPOD(output, this->epsilon);

    // Save index meta-data
    writeBinaryPOD(output, this->mult);

    // Save index state
    writeBinaryPOD(output, this->curElementCount);
    writeBinaryPOD(output, this->numMarkedDeleted);
    writeBinaryPOD(output, this->maxLevel);
    writeBinaryPOD(output, this->entrypointNode);
}

void graphData::restoreGraph(std::ifstream &input) {
	// Restore id to metadata vector
	indexMetaData_.restore(input);
    unsigned int num_blocks = 0;
    readBinaryPOD(input, num_blocks);
	restoreVectorBlocks(input, num_blocksd);
	restoreGraphData(input, num_blocksd);
	
	
}

graphData::saveGraph(std::ofstream &output) const {
    for (idType id = 0; id < this->curElementCount; id++) {
        labelType label = this->idToMetaData[id].label;
        elementFlags flags = this->idToMetaData[id].flags;
        writeBinaryPOD(output, label);
        writeBinaryPOD(output, flags);
    }

    // Save number of blocks
    unsigned int num_blocks = this->vectorBlocks.size();
    writeBinaryPOD(output, num_blocks);

    // Save data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        auto &block = this->vectorBlocks[i];
        unsigned int block_len = block.getLength();
        writeBinaryPOD(output, block_len);
        for (size_t j = 0; j < block_len; j++) {
            output.write(block.getElement(j), this->dataSize);
        }
    }

    // Save graph data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        auto &block = this->graphDataBlocks[i];
        unsigned int block_len = block.getLength();
        writeBinaryPOD(output, block_len);
        for (size_t j = 0; j < block_len; j++) {
            ElementGraphData *cur_element = (ElementGraphData *)block.getElement(j);
            writeBinaryPOD(output, cur_element->toplevel);

            // Save all the levels of the current element
            for (size_t level = 0; level <= cur_element->toplevel; level++) {
                saveLevel(output, getLevelData(cur_element, level));
            }
        }
    }
}

LevelData::saveLevel(LevelData &data) const {
    // Save the links of the current element
    writeBinaryPOD(output, numLinks());
    for (size_t i = 0; i < numLinks(); i++) {
        writeBinaryPOD(output, link(i));
    }

    // Save the incoming edges of the current element
    unsigned int size = incomingEdges()->size();
    writeBinaryPOD(output, size);
    for (idType id : *incomingEdges()) {
        writeBinaryPOD(output, id);
    }

    // Shrink the incoming edges vector for integrity check
    incomingEdges()->shrink_to_fit();
}
#endif
