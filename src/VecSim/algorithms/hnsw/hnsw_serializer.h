#pragma once

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(std::ifstream &input, const HNSWParams *params,
                                         const AbstractIndexInitParams &abstractInitParams,
                                         Serializer::EncodingVersion version)
    : VecSimIndexAbstract<DistType>(abstractInitParams), Serializer(version),
      maxElements(RoundUpInitialCapacity(params->initialCapacity, this->blockSize)),
      epsilon(params->epsilon), vectorBlocks(this->allocator), graphDataBlocks(this->allocator),
      idToMetaData(maxElements, this->allocator),
      visitedNodesHandlerPool(1, maxElements, this->allocator) {

    this->restoreIndexFields(input);
    this->fieldsValidation();

    // Since level generator is implementation-defined, we dont read its value from the file.
    // We use seed = 200 and not the default value (100) to get different sequence of
    // levels value than the loaded index.
    levelGenerator.seed(200);

    size_t initial_vector_size = maxElements / this->blockSize;
    vectorBlocks.reserve(initial_vector_size);
    graphDataBlocks.reserve(initial_vector_size);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexIMP(std::ofstream &output) {
    this->saveIndexFields(output);
    this->saveGraph(output);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::fieldsValidation() const {
    if (this->M > UINT16_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    if (this->M <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1 or 0");
}

template <typename DataType, typename DistType>
HNSWIndexMetaData HNSWIndex<DataType, DistType>::checkIntegrity() const {
    HNSWIndexMetaData res = {.valid_state = false,
                             .memory_usage = -1,
                             .double_connections = HNSW_INVALID_META_DATA,
                             .unidirectional_connections = HNSW_INVALID_META_DATA,
                             .min_in_degree = HNSW_INVALID_META_DATA,
                             .max_in_degree = HNSW_INVALID_META_DATA,
                             .connections_to_repair = 0};

    // Save the current memory usage (before we use additional memory for the integrity check).
    res.memory_usage = this->getAllocationSize();
    size_t connections_checked = 0, double_connections = 0, num_deleted = 0;
    std::vector<int> inbound_connections_num(this->curElementCount, 0);
    size_t incoming_edges_sets_sizes = 0;
    for (size_t i = 0; i < this->curElementCount; i++) {
        if (this->isMarkedDeleted(i)) {
            num_deleted++;
        }
        for (size_t l = 0; l <= getGraphDataByInternalId(i)->toplevel; l++) {
            LevelData &cur = this->getLevelData(i, l);
            std::set<idType> s;
            for (unsigned int j = 0; j < cur.numLinks; j++) {
                // Check if we found an invalid neighbor.
                if (cur.links[j] >= this->curElementCount || cur.links[j] == i) {
                    return res;
                }
                // If the neighbor has deleted, then this connection should be repaired.
                if (isMarkedDeleted(cur.links[j])) {
                    res.connections_to_repair++;
                }
                inbound_connections_num[cur.links[j]]++;
                s.insert(cur.links[j]);
                connections_checked++;

                // Check if this connection is bidirectional.
                LevelData &other = this->getLevelData(cur.links[j], l);
                for (int r = 0; r < other.numLinks; r++) {
                    if (other.links[r] == (idType)i) {
                        double_connections++;
                        break;
                    }
                }
            }
            // Check if a certain neighbor appeared more than once.
            if (s.size() != cur.numLinks) {
                return res;
            }
            incoming_edges_sets_sizes += cur.incomingEdges->size();
        }
    }
    if (num_deleted != this->numMarkedDeleted) {
        return res;
    }
    res.double_connections = double_connections;
    res.unidirectional_connections = incoming_edges_sets_sizes;
    res.min_in_degree =
        !inbound_connections_num.empty()
            ? *std::min_element(inbound_connections_num.begin(), inbound_connections_num.end())
            : 0;
    res.max_in_degree =
        !inbound_connections_num.empty()
            ? *std::max_element(inbound_connections_num.begin(), inbound_connections_num.end())
            : 0;
    if (incoming_edges_sets_sizes + double_connections != connections_checked) {
        return res;
    }

    res.valid_state = true;
    return res;
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreIndexFields(std::ifstream &input) {
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

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreGraph(std::ifstream &input) {
    // Restore id to metadata vector
    labelType label = 0;
    elementFlags flags = 0;
    for (idType id = 0; id < this->curElementCount; id++) {
        readBinaryPOD(input, label);
        readBinaryPOD(input, flags);
        this->idToMetaData[id].label = label;
        this->idToMetaData[id].flags = flags;

        // Restore label lookup by getting the label from data_level0_memory_
        setVectorId(label, id);
    }

    // Get number of blocks
    unsigned int num_blocks = 0;
    readBinaryPOD(input, num_blocks);
    this->vectorBlocks.reserve(num_blocks);
    this->graphDataBlocks.reserve(num_blocks);

    // Get data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        this->vectorBlocks.emplace_back(this->blockSize, this->dataSize, this->allocator,
                                        this->alignment);
        unsigned int block_len = 0;
        readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            char cur_vec[this->dataSize];
            input.read(cur_vec, this->dataSize);
            this->vectorBlocks.back().addElement(cur_vec);
        }
    }

    // Get graph data blocks
    ElementGraphData *cur_egt;
    char tmpData[this->elementGraphDataSize];
    size_t toplevel = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        this->graphDataBlocks.emplace_back(this->blockSize, this->elementGraphDataSize,
                                           this->allocator);
        unsigned int block_len = 0;
        readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            // Reset tmpData
            memset(tmpData, 0, this->elementGraphDataSize);
            // Read the current element top level
            readBinaryPOD(input, toplevel);
            // Allocate space and structs for the current element
            try {
                new (tmpData) ElementGraphData(toplevel, this->levelDataSize, this->allocator);
            } catch (std::runtime_error &e) {
                this->log("Error - allocating memory for new element failed due to low memory");
                throw e;
            }
            // Add the current element to the current block, and update cur_egt to point to it.
            this->graphDataBlocks.back().addElement(tmpData);
            cur_egt = (ElementGraphData *)this->graphDataBlocks.back().getElement(j);

            // Restore the current element's graph data
            for (size_t k = 0; k <= toplevel; k++) {
                restoreLevel(input, getLevelData(cur_egt, k));
            }
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreLevel(std::ifstream &input, LevelData &data) {
    // Restore the links of the current element
    readBinaryPOD(input, data.numLinks);
    for (size_t i = 0; i < data.numLinks; i++) {
        readBinaryPOD(input, data.links[i]);
    }

    // Restore the incoming edges of the current element
    unsigned int size;
    readBinaryPOD(input, size);
    data.incomingEdges->reserve(size);
    idType id = INVALID_ID;
    for (size_t i = 0; i < size; i++) {
        readBinaryPOD(input, id);
        data.incomingEdges->push_back(id);
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexFields(std::ofstream &output) const {
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

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveGraph(std::ofstream &output) const {
    // Save id to metadata vector
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

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveLevel(std::ofstream &output, LevelData &data) const {
    // Save the links of the current element
    writeBinaryPOD(output, data.numLinks);
    for (size_t i = 0; i < data.numLinks; i++) {
        writeBinaryPOD(output, data.links[i]);
    }

    // Save the incoming edges of the current element
    unsigned int size = data.incomingEdges->size();
    writeBinaryPOD(output, size);
    for (idType id : *data.incomingEdges) {
        writeBinaryPOD(output, id);
    }

    // Shrink the incoming edges vector for integrity check
    data.incomingEdges->shrink_to_fit();
}
