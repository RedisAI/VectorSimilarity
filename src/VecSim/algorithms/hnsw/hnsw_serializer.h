#pragma once

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(std::ifstream &input, const HNSWParams *params,
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         EncodingVersion version)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric,
                                    params->blockSize, params->multi),
      VecSimIndexTombstone(), Serializer(version), max_elements_(params->initialCapacity),
      epsilon_(params->epsilon), vector_blocks(allocator), meta_blocks(allocator),
      idToMetaData(max_elements_, allocator) {

    this->restoreIndexFields(input);
    this->fieldsValidation();

    // Since level generator is implementation-defined, we dont read its value from the file.
    // We use seed = 200 and not the default value (100) to get different sequence of
    // levels value than the loaded index.
    level_generator_.seed(200);

#ifdef ENABLE_PARALLELIZATION
    this->pool_initial_size = 1;
    this->visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(
        new (this->allocator)
            VisitedNodesHandlerPool(this->pool_initial_size, max_elements_, this->allocator));
#else
    this->visited_nodes_handler = std::unique_ptr<VisitedNodesHandler>(
        new (this->allocator) VisitedNodesHandler(max_elements_, this->allocator));
#endif
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
                             .max_in_degree = HNSW_INVALID_META_DATA};

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

    // epsilon is only restored from v2 up.

    // Restore index meta-data
    readBinaryPOD(input, this->element_data_size_);
    readBinaryPOD(input, this->element_graph_data_size_);
    readBinaryPOD(input, this->level_data_size_);
    // readBinaryPOD(input, this->data_size_);
    // readBinaryPOD(input, this->size_data_per_element_);
    // readBinaryPOD(input, this->size_links_per_element_);
    // readBinaryPOD(input, this->size_links_level0_);
    // readBinaryPOD(input, this->label_offset_);
    // readBinaryPOD(input, this->offsetData_);
    // readBinaryPOD(input, this->offsetLevel0_);
    // readBinaryPOD(input, this->incoming_links_offset0);
    // readBinaryPOD(input, this->incoming_links_offset);
    readBinaryPOD(input, this->mult_);

    // skip restoration of level_generator_ data member
    HandleLevelGenerator(input);

    // Restore index state
    readBinaryPOD(input, this->cur_element_count);
    if (this->m_version == EncodingVersion_V1) {
        input.ignore(sizeof(idType)); // skip max_id value
        this->num_marked_deleted = 0;
    } else {
        readBinaryPOD(input, this->num_marked_deleted);
    }
    readBinaryPOD(input, this->maxlevel_);
    readBinaryPOD(input, this->entrypoint_node_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::HandleLevelGenerator(std::ifstream &input) {
    if (this->m_version == EncodingVersion_V1) {
        // All current v1 files were generated on intel machines, where
        // sizeof(std::default_random_engine) ==  sizeof(unsigned long)
        // unlike MacOS where sizeof(std::default_random_engine) ==  sizeof(unsigned int).

        // Skip sizeof(unsigned long) bytes
        input.ignore(sizeof(unsigned long));
    }
    // for V2 and up we don't serialize the level generator, so we just return and
    // continue to read the file.
}

// template <typename DataType, typename DistType>
// void HNSWIndex<DataType, DistType>::restoreGraph_V1_fixes() {
//     // Fix offsets from V1 to V2
//     size_t old_size_links_per_element_ = this->size_links_per_element_;
//     this->size_links_per_element_ -= sizeof(idType) - sizeof(linkListSize);
//     this->incoming_links_offset -= sizeof(idType) - sizeof(linkListSize);

//     char *data = this->data_level0_memory_;
//     for (idType i = 0; i < this->cur_element_count; i++) {
//         // Restore level 0 number of links
//         // In V1 linkListSize was of the same size as idType, so we need to fix it.
//         // V1 did not have the elementFlags, so we need set all flags to 0.
//         idType lls = *(idType *)data;
//         *(linkListSize *)(data + sizeof(elementFlags)) = (linkListSize)lls;
//         *(elementFlags *)(data) = (elementFlags)0;
//         data += this->size_data_per_element_;

//         // Restore level 1+ links
//         // We need to fix the offset of the linkListSize.
//         size_t llSize = this->element_levels_[i] * this->size_links_per_element_;
//         if (llSize) {
//             char *levels_data = (char *)this->allocator->allocate(llSize);
//             for (size_t offset = 0; offset < this->element_levels_[i]; offset++) {
//                 // Copy links without the linkListSize
//                 // sizeof(linkListSize) == New offset size
//                 // sizeof(idType) == Old offset size
//                 memcpy(levels_data + offset * this->size_links_per_element_ +
//                 sizeof(linkListSize),
//                        this->linkLists_[i] + offset * old_size_links_per_element_ +
//                        sizeof(idType), this->size_links_per_element_ - sizeof(linkListSize));
//                 // Copy linkListSize (from idType to linkListSize)
//                 *(linkListSize *)(levels_data + offset * this->size_links_per_element_) =
//                     *(idType *)(this->linkLists_[i] + offset * old_size_links_per_element_);
//             }
//             // Free old links and set new links
//             this->allocator->free_allocation(this->linkLists_[i]);
//             this->linkLists_[i] = levels_data;
//         }
//     }
// }

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreGraph(std::ifstream &input) {
    // Restore id to metadata vector
    element_meta_data cur_meta;
    for (idType id = 0; id < this->cur_element_count; id++) {
        readBinaryPOD(input, cur_meta);
        this->idToMetaData[id] = cur_meta;

        // Restore label lookup by getting the label from data_level0_memory_
        setVectorId(cur_meta.label, id);
    }

    // Get number of blocks
    unsigned int num_blocks = 0;
    readBinaryPOD(input, num_blocks);
    this->vector_blocks.reserve(num_blocks);
    this->meta_blocks.reserve(num_blocks);

    // Get data blocks
    for (size_t i = 0; i < num_blocks; i++) {
        this->vector_blocks.emplace_back(this->blockSize, this->element_data_size_,
                                         this->allocator);
        unsigned int block_len = 0;
        readBinaryPOD(input, block_len);
        for (size_t j = 0; j < block_len; j++) {
            char cur_vec[this->element_data_size_];
            input.read(cur_vec, this->element_data_size_);
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
    if (this->m_version == EncodingVersion_V1) {
        // restoreGraph_V1_fixes();
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexFields_v2(std::ofstream &output) const {
    // From v2 and up write also algorithm type and vec_sim_index data members.
    writeBinaryPOD(output, VecSimAlgo_HNSWLIB);

    writeBinaryPOD(output, this->dim);
    writeBinaryPOD(output, this->vecType);
    writeBinaryPOD(output, this->metric);
    writeBinaryPOD(output, this->blockSize);
    writeBinaryPOD(output, this->isMulti);
    writeBinaryPOD(output, this->epsilon_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexFields(std::ofstream &output) const {

    this->saveIndexFields_v2(output);

    // Save index build parameters
    writeBinaryPOD(output, this->max_elements_);
    writeBinaryPOD(output, this->M_);
    writeBinaryPOD(output, this->maxM_);
    writeBinaryPOD(output, this->maxM0_);
    writeBinaryPOD(output, this->ef_construction_);

    // Save index search parameter
    writeBinaryPOD(output, this->ef_);

    // Save index meta-data
    // writeBinaryPOD(output, this->data_size_);
    // writeBinaryPOD(output, this->size_data_per_element_);
    // writeBinaryPOD(output, this->size_links_per_element_);
    // writeBinaryPOD(output, this->size_links_level0_);
    // writeBinaryPOD(output, this->label_offset_);
    // writeBinaryPOD(output, this->offsetData_);
    // writeBinaryPOD(output, this->offsetLevel0_);
    // writeBinaryPOD(output, this->incoming_links_offset0);
    // writeBinaryPOD(output, this->incoming_links_offset);
    writeBinaryPOD(output, this->element_data_size_);
    writeBinaryPOD(output, this->element_graph_data_size_);
    writeBinaryPOD(output, this->level_data_size_);
    writeBinaryPOD(output, this->mult_);

    // Save index state
    writeBinaryPOD(output, this->cur_element_count);
    writeBinaryPOD(output, this->num_marked_deleted);
    writeBinaryPOD(output, this->maxlevel_);
    writeBinaryPOD(output, this->entrypoint_node_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveGraph(std::ofstream &output) const {
    // Save id to metadata vector
    for (idType id = 0; id < this->cur_element_count; id++) {
        writeBinaryPOD(output, this->idToMetaData[id]);
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
            output.write(block.getElement(j), this->element_data_size_);
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
