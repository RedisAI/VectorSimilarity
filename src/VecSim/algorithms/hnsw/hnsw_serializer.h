#pragma once

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(std::ifstream &input, const HNSWParams *params,
                                         const AbstractIndexInitParams &abstractInitParams,
                                         Serializer::EncodingVersion version)
    : VecSimIndexAbstract<DistType>(abstractInitParams), Serializer(version),
      max_elements_(params->initialCapacity), epsilon_(params->epsilon),
      vector_blocks(this->allocator), meta_blocks(this->allocator), idToMetaData(this->allocator),
      visited_nodes_handler_pool(1, max_elements_, this->allocator)
{

    this->restoreIndexFields(input);
    this->fieldsValidation();

    // Since level generator is implementation-defined, we dont read its value from the file.
    // We use seed = 200 and not the default value (100) to get different sequence of
    // levels value than the loaded index.
    level_generator_.seed(200);

    size_t initial_vector_size = max_elements_ / this->blockSize;
    if (max_elements_ % this->blockSize != 0) {
        initial_vector_size++;
    }
    vector_blocks.reserve(initial_vector_size);
    meta_blocks.reserve(initial_vector_size);

    idToMetaData.resize(max_elements_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexIMP(std::ofstream &output) {
    this->saveIndexFields(output);
    this->saveGraph(output);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::fieldsValidation() const {
    if (this->M_ > UINT16_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    if (this->M_ <= 1)
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
    std::vector<int> inbound_connections_num(this->cur_element_count, 0);
    size_t incoming_edges_sets_sizes = 0;
    for (size_t i = 0; i < this->cur_element_count; i++) {
        if (this->isMarkedDeleted(i)) {
            num_deleted++;
        }
        for (size_t l = 0; l <= getMetaDataByInternalId(i)->toplevel; l++) {
            level_data &cur = this->getLevelData(i, l);
            std::set<idType> s;
            for (unsigned int j = 0; j < cur.numLinks; j++) {
                // Check if we found an invalid neighbor.
                if (cur.links[j] >= this->cur_element_count || cur.links[j] == i) {
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
                level_data &other = this->getLevelData(cur.links[j], l);
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
            incoming_edges_sets_sizes += cur.incoming_edges->size();
        }
    }
    if (num_deleted != this->num_marked_deleted) {
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
    readBinaryPOD(input, this->M_);
    readBinaryPOD(input, this->maxM_);
    readBinaryPOD(input, this->maxM0_);
    readBinaryPOD(input, this->ef_construction_);

    // Restore index search parameter
    readBinaryPOD(input, this->ef_);
    readBinaryPOD(input, this->epsilon_);

    // Restore index meta-data
    readBinaryPOD(input, this->element_graph_data_size_);
    readBinaryPOD(input, this->level_data_size_);
    readBinaryPOD(input, this->mult_);

    // Restore index state
    readBinaryPOD(input, this->cur_element_count);
    readBinaryPOD(input, this->num_marked_deleted);
    readBinaryPOD(input, this->max_level_);
    readBinaryPOD(input, this->entrypoint_node_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreGraph(std::ifstream &input) {
    // Restore id to metadata vector
    labelType label = 0;
    elementFlags flags = 0;
    for (idType id = 0; id < this->cur_element_count; id++) {
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
    this->vector_blocks.reserve(num_blocks);
    this->meta_blocks.reserve(num_blocks);

    // Get data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        this->vector_blocks.emplace_back(this->blockSize, this->data_size, this->allocator);
        unsigned int block_len = 0;
        readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            char cur_vec[this->data_size];
            input.read(cur_vec, this->data_size);
            this->vector_blocks.back().addElement(cur_vec);
        }
    }

    // Get meta blocks
    idType cur_c = 0;
    for (size_t i = 0; i < num_blocks; i++) {
        this->meta_blocks.emplace_back(this->blockSize, this->element_graph_data_size_,
                                       this->allocator);
        unsigned int block_len = 0;
        readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            char cur_meta_data[this->element_graph_data_size_];
            input.read(cur_meta_data, this->element_graph_data_size_);
            auto cur_meta = (element_graph_data *)cur_meta_data;

            if (cur_meta->toplevel > 0) {
                // Allocate space for the other levels
                cur_meta->others = (level_data *)this->allocator->allocate(this->level_data_size_ *
                                                                           cur_meta->toplevel);
                if (cur_meta->others == nullptr) {
                    throw std::runtime_error(
                        "Not enough memory: loadIndex failed to allocate element meta data.");
                }
                input.read((char *)(cur_meta->others), this->level_data_size_ * cur_meta->toplevel);
            }

            // Save the incoming edges of the current element.
            // Level 0
            unsigned int size = 0;
            readBinaryPOD(input, size);
            cur_meta->level0.incoming_edges =
                new (this->allocator) vecsim_stl::vector<idType>(size, this->allocator);
            for (size_t k = 0; k < size; k++) {
                idType edge;
                readBinaryPOD(input, edge);
                (*cur_meta->level0.incoming_edges)[k] = edge;
            }

            // Levels 1 to maxlevel
            for (size_t level_offset = 0; level_offset < cur_meta->toplevel; level_offset++) {
                auto cur = (level_data *)(((char *)cur_meta->others) +
                                          level_offset * this->level_data_size_);
                unsigned int size = 0;
                readBinaryPOD(input, size);
                cur->incoming_edges =
                    new (this->allocator) vecsim_stl::vector<idType>(size, this->allocator);
                for (size_t k = 0; k < size; k++) {
                    idType edge;
                    readBinaryPOD(input, edge);
                    (*cur->incoming_edges)[k] = edge;
                }
            }

            this->meta_blocks.back().addElement(cur_meta_data);
            cur_c++;
        }
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
    writeBinaryPOD(output, this->max_elements_); // This will be used to restore the index initial
                                                 // capacity

    // Save index build parameters
    writeBinaryPOD(output, this->M_);
    writeBinaryPOD(output, this->maxM_);
    writeBinaryPOD(output, this->maxM0_);
    writeBinaryPOD(output, this->ef_construction_);

    // Save index search parameter
    writeBinaryPOD(output, this->ef_);
    writeBinaryPOD(output, this->epsilon_);

    // Save index meta-data
    writeBinaryPOD(output, this->element_graph_data_size_);
    writeBinaryPOD(output, this->level_data_size_);
    writeBinaryPOD(output, this->mult_);

    // Save index state
    writeBinaryPOD(output, this->cur_element_count);
    writeBinaryPOD(output, this->num_marked_deleted);
    writeBinaryPOD(output, this->max_level_);
    writeBinaryPOD(output, this->entrypoint_node_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveGraph(std::ofstream &output) const {
    // Save id to metadata vector
    for (idType id = 0; id < this->cur_element_count; id++) {
        labelType label = this->idToMetaData[id].label;
        elementFlags flags = this->idToMetaData[id].flags;
        writeBinaryPOD(output, label);
        writeBinaryPOD(output, flags);
    }

    // Save number of blocks
    unsigned int num_blocks = this->vector_blocks.size();
    writeBinaryPOD(output, num_blocks);

    // Save data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        auto &block = this->vector_blocks[i];
        unsigned int block_len = block.getLength();
        writeBinaryPOD(output, block_len);
        for (size_t j = 0; j < block_len; j++) {
            output.write(block.getElement(j), this->data_size);
        }
    }

    // Save meta blocks
    for (size_t i = 0; i < num_blocks; i++) {
        auto &block = this->meta_blocks[i];
        unsigned int block_len = block.getLength();
        writeBinaryPOD(output, block_len);
        for (size_t j = 0; j < block_len; j++) {
            element_graph_data *meta = (element_graph_data *)block.getElement(j);
            output.write((char *)meta, this->element_graph_data_size_);
            if (meta->others) // only if there are levels > 0
                output.write((char *)meta->others, this->level_data_size_ * meta->toplevel);

            // Save the incoming edges of the current element.
            // Level 0
            unsigned int size = meta->level0.incoming_edges->size();
            writeBinaryPOD(output, size);
            for (idType id : *meta->level0.incoming_edges) {
                writeBinaryPOD(output, id);
            }
            meta->level0.incoming_edges->shrink_to_fit();

            // Levels 1 to maxlevel
            for (size_t level_offset = 0; level_offset < meta->toplevel; level_offset++) {
                auto cur =
                    (level_data *)(((char *)meta->others) + level_offset * this->level_data_size_);
                unsigned int size = cur->incoming_edges->size();
                writeBinaryPOD(output, size);
                for (idType id : *cur->incoming_edges) {
                    writeBinaryPOD(output, id);
                }
                cur->incoming_edges->shrink_to_fit();
            }
        }
    }
}
