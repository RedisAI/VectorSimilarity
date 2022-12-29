#pragma once

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(std::ifstream &input, const HNSWParams *params,
                                         std::shared_ptr<VecSimAllocator> allocator,
                                         EncodingVersion version)
    : VecSimIndexAbstract<DistType>(allocator, params->dim, params->type, params->metric,
                                    params->blockSize, params->multi),
      Serializer(version), max_elements_(params->initialCapacity), epsilon_(params->epsilon),
      element_levels_(max_elements_, allocator),
      visited_nodes_handler_pool(1, max_elements_, allocator) {

    this->restoreIndexFields(input);
    this->fieldsValidation();

    // Since level generator is implementation-defined, we dont read its value from the file.
    // We use seed = 200 and not the default value (100) to get different sequence of
    // levels value than the loaded index.
    level_generator_.seed(200);

    data_level0_memory_ =
        (char *)this->allocator->callocate(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
        throw std::runtime_error("Not enough memory");

    linkLists_ = (char **)this->allocator->callocate(sizeof(void *) * max_elements_);
    if (linkLists_ == nullptr)
        throw std::runtime_error("Not enough memory: HNSWIndex failed to allocate linklists");
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
        for (size_t l = 0; l <= this->element_levels_[i]; l++) {
            idType *cur_links = this->get_linklist_at_level(i, l);
            linkListSize size = this->getListCount(cur_links);
            std::set<idType> s;
            for (unsigned int j = 0; j < size; j++) {
                // Check if we found an invalid neighbor.
                if (cur_links[j] >= this->cur_element_count || cur_links[j] == i) {
                    return res;
                }
                inbound_connections_num[cur_links[j]]++;
                s.insert(cur_links[j]);
                connections_checked++;

                // Check if this connection is bidirectional.
                idType *other_links = this->get_linklist_at_level(cur_links[j], l);
                linkListSize size_other = this->getListCount(other_links);
                for (int r = 0; r < size_other; r++) {
                    if (other_links[r] == (idType)i) {
                        double_connections++;
                        break;
                    }
                }
            }
            // Check if a certain neighbor appeared more than once.
            if (s.size() != size) {
                return res;
            }
            incoming_edges_sets_sizes += this->getIncomingEdgesPtr(i, l)->size();
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
    readBinaryPOD(input, this->data_size_);
    readBinaryPOD(input, this->size_data_per_element_);
    readBinaryPOD(input, this->size_links_per_element_);
    readBinaryPOD(input, this->size_links_level0_);
    readBinaryPOD(input, this->label_offset_);
    readBinaryPOD(input, this->offsetData_);
    readBinaryPOD(input, this->offsetLevel0_);
    readBinaryPOD(input, this->incoming_links_offset0);
    readBinaryPOD(input, this->incoming_links_offset);
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

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreGraph_V1_fixes() {
    // Fix offsets from V1 to V2
    size_t old_size_links_per_element_ = this->size_links_per_element_;
    this->size_links_per_element_ -= sizeof(idType) - sizeof(linkListSize);
    this->incoming_links_offset -= sizeof(idType) - sizeof(linkListSize);

    char *data = this->data_level0_memory_;
    for (idType i = 0; i < this->cur_element_count; i++) {
        // Restore level 0 number of links
        // In V1 linkListSize was of the same size as idType, so we need to fix it.
        // V1 did not have the elementFlags, so we need set all flags to 0.
        idType lls = *(idType *)data;
        *(linkListSize *)(data + sizeof(elementFlags)) = (linkListSize)lls;
        *(elementFlags *)(data) = (elementFlags)0;
        data += this->size_data_per_element_;

        // Restore level 1+ links
        // We need to fix the offset of the linkListSize.
        size_t llSize = this->element_levels_[i] * this->size_links_per_element_;
        if (llSize) {
            char *levels_data = (char *)this->allocator->allocate(llSize);
            for (size_t offset = 0; offset < this->element_levels_[i]; offset++) {
                // Copy links without the linkListSize
                // sizeof(linkListSize) == New offset size
                // sizeof(idType) == Old offset size
                memcpy(levels_data + offset * this->size_links_per_element_ + sizeof(linkListSize),
                       this->linkLists_[i] + offset * old_size_links_per_element_ + sizeof(idType),
                       this->size_links_per_element_ - sizeof(linkListSize));
                // Copy linkListSize (from idType to linkListSize)
                *(linkListSize *)(levels_data + offset * this->size_links_per_element_) =
                    *(idType *)(this->linkLists_[i] + offset * old_size_links_per_element_);
            }
            // Free old links and set new links
            this->allocator->free_allocation(this->linkLists_[i]);
            this->linkLists_[i] = levels_data;
        }
    }
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreGraph(std::ifstream &input) {
    // Restore graph layer 0
    input.read(this->data_level0_memory_, this->max_elements_ * this->size_data_per_element_);
    for (idType i = 0; i < this->cur_element_count; i++) {
        auto *incoming_edges = new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
        unsigned int incoming_edges_len;
        readBinaryPOD(input, incoming_edges_len);
        for (size_t j = 0; j < incoming_edges_len; j++) {
            idType next_edge;
            readBinaryPOD(input, next_edge);
            incoming_edges->push_back(next_edge);
        }
        incoming_edges->shrink_to_fit();
        this->setIncomingEdgesPtr(i, 0, (void *)incoming_edges);
    }
    // Restore the rest of the graph layers, along with the label and max_level lookups.
    for (idType i = 0; i < this->cur_element_count; i++) {
        // Restore label lookup by getting the label from data_level0_memory_
        setVectorId(getExternalLabel(i), i);

        linkListSize linkList_size;
        if (this->m_version == EncodingVersion_V1) {
            idType lls;
            readBinaryPOD(input, lls);
            linkList_size = (linkListSize)lls;
        } else {
            readBinaryPOD(input, linkList_size);
        }

        if (linkList_size == 0) {
            this->element_levels_[i] = 0;
            this->linkLists_[i] = nullptr;
        } else {
            this->element_levels_[i] = linkList_size / this->size_links_per_element_;
            this->linkLists_[i] = (char *)this->allocator->allocate(linkList_size);
            if (this->linkLists_[i] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist");
            input.read(this->linkLists_[i], linkList_size);
            for (size_t j = 1; j <= this->element_levels_[i]; j++) {
                auto *incoming_edges =
                    new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
                unsigned int vector_len;
                readBinaryPOD(input, vector_len);
                for (size_t k = 0; k < vector_len; k++) {
                    idType next_edge;
                    readBinaryPOD(input, next_edge);
                    incoming_edges->push_back(next_edge);
                }
                incoming_edges->shrink_to_fit();
                this->setIncomingEdgesPtr(i, j, (void *)incoming_edges);
            }
        }
    }
    if (this->m_version == EncodingVersion_V1) {
        restoreGraph_V1_fixes();
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
    writeBinaryPOD(output, this->data_size_);
    writeBinaryPOD(output, this->size_data_per_element_);
    writeBinaryPOD(output, this->size_links_per_element_);
    writeBinaryPOD(output, this->size_links_level0_);
    writeBinaryPOD(output, this->label_offset_);
    writeBinaryPOD(output, this->offsetData_);
    writeBinaryPOD(output, this->offsetLevel0_);
    writeBinaryPOD(output, this->incoming_links_offset0);
    writeBinaryPOD(output, this->incoming_links_offset);
    writeBinaryPOD(output, this->mult_);

    // Save index state
    writeBinaryPOD(output, this->cur_element_count);
    writeBinaryPOD(output, this->num_marked_deleted);
    writeBinaryPOD(output, this->maxlevel_);
    writeBinaryPOD(output, this->entrypoint_node_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveGraph(std::ofstream &output) const {
    // Save level 0 data (graph layer 0 + labels + vectors data)
    output.write(this->data_level0_memory_, this->max_elements_ * this->size_data_per_element_);

    // Save the incoming edge sets.
    for (size_t i = 0; i < this->cur_element_count; i++) {
        auto *incoming_edges_ptr = this->getIncomingEdgesPtr(i, 0);
        unsigned int set_size = incoming_edges_ptr->size();
        writeBinaryPOD(output, set_size);
        for (auto id : *incoming_edges_ptr) {
            writeBinaryPOD(output, id);
        }
        incoming_edges_ptr->shrink_to_fit();
    }

    // Save all graph layers other than layer 0: for every id of a vector in the graph,
    // store (<size>, data), where <size> is the data size, and the data is the concatenated
    // adjacency lists in the graph Then, store the sets of the incoming edges in every level.
    for (size_t i = 0; i < this->cur_element_count; i++) {
        linkListSize linkList_size = this->element_levels_[i] > 0
                                         ? this->size_links_per_element_ * this->element_levels_[i]
                                         : 0;
        writeBinaryPOD(output, linkList_size);
        if (linkList_size)
            output.write(this->linkLists_[i], linkList_size);
        for (size_t j = 1; j <= this->element_levels_[i]; j++) {
            auto *incoming_edges_ptr = this->getIncomingEdgesPtr(i, j);
            unsigned int set_size = incoming_edges_ptr->size();
            writeBinaryPOD(output, set_size);
            for (auto id : *incoming_edges_ptr) {
                writeBinaryPOD(output, id);
            }
            incoming_edges_ptr->shrink_to_fit();
        }
    }
}
