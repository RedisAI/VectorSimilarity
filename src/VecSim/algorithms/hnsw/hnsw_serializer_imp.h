#pragma once

template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(std::string location,
                                         std::shared_ptr<VecSimAllocator> allocator)
    : VecSimIndexAbstract<DistType>(allocator), element_levels_(0, allocator) {
    VecSimType type = this->vecType;
    bool is_multi = this->isMulti;

    this->loadIndex(location);
    if (type != this->vecType || is_multi != this->isMulti) {
        throw std::runtime_error("Wrong type index");
    }
}
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexIMP(std::ofstream &output,
                                                 EncodingVersion version) const {
    // We already checked in the serializer that this is a valid version number.
    // Now checking the version number to decide which data to write.
    if (version != EncodingVersion_V1) {
        saveIndexFields_v2(output);
    }

    this->saveIndexFields(output);
    this->saveGraph(output);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::loadIndexIMP(std::ifstream &input, EncodingVersion version) {
    // We already checked in the serializer that this is a valid version number.
    if (version != EncodingVersion_V1) {
        // This data is only serialized from V2 up.
        VecSimAlgo algo = VecSimAlgo_INVALID;
        readBinaryPOD(input, algo);
        if (algo != VecSimAlgo_HNSWLIB) {
            input.close();
            throw std::runtime_error("Cannot load index: bad algorithm type");
        }
        restoreIndexFields_v2(input);
    }

    this->restoreIndexFields(input);

    // Resize all data structure to the serialized index memory sizes.
    clearLabelLookup();

    resizeIndex(this->max_elements_);
    this->restoreGraph(input);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::fieldsValidation() const {
    if (this->M_ > SIZE_MAX / 2)
        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    if (this->M_ <= 1)
        throw std::runtime_error("HNSW index parameter M cannot be 1 or 0");
    if (this->maxM0_ >
            ((SIZE_MAX - sizeof(void *) - sizeof(linklistsizeint)) / sizeof(idType)) + 1 ||
        this->size_links_level0_ > SIZE_MAX - data_size_ - sizeof(labelType)) {

        throw std::runtime_error("HNSW index parameter M is too large: argument overflow");
    }
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
    size_t connections_checked = 0, double_connections = 0;
    std::vector<int> inbound_connections_num(this->max_id + 1, 0);
    size_t incoming_edges_sets_sizes = 0;
    if (this->max_id != HNSW_INVALID_ID) {
        for (size_t i = 0; i <= this->max_id; i++) {
            for (size_t l = 0; l <= this->element_levels_[i]; l++) {
                linklistsizeint *ll_cur = this->get_linklist_at_level(i, l);
                unsigned int size = this->getListCount(ll_cur);
                auto *data = (idType *)(ll_cur + 1);
                std::set<idType> s;
                for (unsigned int j = 0; j < size; j++) {
                    // Check if we found an invalid neighbor.
                    if (data[j] > this->max_id || data[j] == i) {
                        res.valid_state = false;
                        return res;
                    }
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;

                    // Check if this connection is bidirectional.
                    linklistsizeint *ll_other = this->get_linklist_at_level(data[j], l);
                    int size_other = this->getListCount(ll_other);
                    auto *data_other = (idType *)(ll_other + 1);
                    for (int r = 0; r < size_other; r++) {
                        if (data_other[r] == (idType)i) {
                            double_connections++;
                            break;
                        }
                    }
                }
                // Check if a certain neighbor appeared more than once.
                if (s.size() != size) {
                    res.valid_state = false;
                    return res;
                }
                incoming_edges_sets_sizes += this->getIncomingEdgesPtr(i, l)->size();
            }
        }
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
        res.valid_state = false;
        return res;
    }
    res.valid_state = true;
    return res;
}
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreIndexFields_v2(std::ifstream &input) {

    readBinaryPOD(input, this->dim);
    readBinaryPOD(input, this->vecType);
    readBinaryPOD(input, this->metric);
    readBinaryPOD(input, this->blockSize);
    readBinaryPOD(input, this->dist_func);
    readBinaryPOD(input, this->isMulti);
    readBinaryPOD(input, this->epsilon_);
}
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreIndexFields(std::ifstream &input) {
    // Restore index build parameters
    readBinaryPOD(input, this->max_elements_);
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

    // Restore index level generator of the top level for a new element
    readBinaryPOD(input, this->level_generator_);

    // Restore index state
    readBinaryPOD(input, this->cur_element_count);
    readBinaryPOD(input, this->max_id);
    readBinaryPOD(input, this->maxlevel_);
    readBinaryPOD(input, this->entrypoint_node_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::restoreGraph(std::ifstream &input) {
    // Restore graph layer 0
    input.read(this->data_level0_memory_, this->max_elements_ * this->size_data_per_element_);
    if (this->max_id == HNSW_INVALID_ID) {
        return; // Index is empty.
    }
    for (size_t i = 0; i <= this->max_id; i++) {
        auto *incoming_edges = new (this->allocator) vecsim_stl::vector<idType>(this->allocator);
        unsigned int incoming_edges_len;
        readBinaryPOD(input, incoming_edges_len);
        for (size_t j = 0; j < incoming_edges_len; j++) {
            idType next_edge;
            readBinaryPOD(input, next_edge);
            incoming_edges->push_back(next_edge);
        }
        this->setIncomingEdgesPtr(i, 0, (void *)incoming_edges);
    }

    // Restore the rest of the graph layers, along with the label and max_level lookups.
    for (size_t i = 0; i <= this->max_id; i++) {
        // Restore label lookup by getting the label from data_level0_memory_
        AddToLabelLookup(getExternalLabel(i), i);

        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize == 0) {
            this->element_levels_[i] = 0;
            this->linkLists_[i] = nullptr;
        } else {
            this->element_levels_[i] = linkListSize / this->size_links_per_element_;
            this->linkLists_[i] = (char *)this->allocator->allocate(linkListSize);
            if (this->linkLists_[i] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist");
            input.read(this->linkLists_[i], linkListSize);
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
                this->setIncomingEdgesPtr(i, j, (void *)incoming_edges);
            }
        }
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
    writeBinaryPOD(output, this->dist_func);
    writeBinaryPOD(output, this->isMulti);
    writeBinaryPOD(output, this->epsilon_);
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexFields(std::ofstream &output) const {
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

    // Save index level generator of the top level for a new element
    writeBinaryPOD(output, this->level_generator_);

    // Save index state
    writeBinaryPOD(output, this->cur_element_count);
    writeBinaryPOD(output, this->max_id);
    writeBinaryPOD(output, this->maxlevel_);
    writeBinaryPOD(output, this->entrypoint_node_);
}
template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveGraph(std::ofstream &output) const {
    // Save level 0 data (graph layer 0 + labels + vectors data)
    output.write(this->data_level0_memory_, this->max_elements_ * this->size_data_per_element_);
    if (this->max_id == HNSW_INVALID_ID) {
        return; // Index is empty.
    }
    // Save the incoming edge sets.
    for (size_t i = 0; i <= this->max_id; i++) {
        auto *incoming_edges_ptr = this->getIncomingEdgesPtr(i, 0);
        unsigned int set_size = incoming_edges_ptr->size();
        writeBinaryPOD(output, set_size);
        for (auto id : *incoming_edges_ptr) {
            writeBinaryPOD(output, id);
        }
    }

    // Save all graph layers other than layer 0: for every id of a vector in the graph,
    // store (<size>, data), where <size> is the data size, and the data is the concatenated
    // adjacency lists in the graph Then, store the sets of the incoming edges in every level.
    for (size_t i = 0; i <= this->max_id; i++) {
        unsigned int linkListSize = this->element_levels_[i] > 0
                                        ? this->size_links_per_element_ * this->element_levels_[i]
                                        : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize)
            output.write(this->linkLists_[i], linkListSize);
        for (size_t j = 1; j <= this->element_levels_[i]; j++) {
            auto *incoming_edges_ptr = this->getIncomingEdgesPtr(i, j);
            unsigned int set_size = incoming_edges_ptr->size();
            writeBinaryPOD(output, set_size);
            for (auto id : *incoming_edges_ptr) {
                writeBinaryPOD(output, id);
            }
        }
    }
}
