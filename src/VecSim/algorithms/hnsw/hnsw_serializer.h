#pragma once
template <typename DataType, typename DistType>
HNSWIndex<DataType, DistType>::HNSWIndex(std::ifstream &input, const HNSWParams *params,
                                         const AbstractIndexInitParams &abstractInitParams,
                                         Serializer::EncodingVersion version)
    : VecSimIndexAbstract<DataType, DistType>(abstractInitParams), Serializer(version),
	  graphData(this->allocator, *this),
      visitedNodesHandlerPool(1, RoundUpInitialCapacity(params->initialCapacity,
								   this->blockSize), this->allocator) {
	
	maxElements = RoundUpInitialCapacity(params->initialCapacity, this->blockSize);
	epsilon = params->epsilon;
	blockSize_ = this->blockSize;
	alignment_ = this->alignment;
	vectorDataSize = this->dataSize;

	IndexMetaData::restore(input);
    this->fieldsValidation();

    // Since level generator is implementation-defined, we dont read its value from the file.
    // We use seed = 200 and not the default value (100) to get different sequence of
    // levels value than the loaded index.
    levelGenerator.seed(200);

	graphData.Init();
	
}

template <typename DataType, typename DistType>
void HNSWIndex<DataType, DistType>::saveIndexIMP(std::ofstream &output) {
	graphData.save(output);
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
            for (unsigned int j = 0; j < cur.numLinks(); j++) {
                // Check if we found an invalid neighbor.
                if (cur.link(j) >= this->curElementCount || cur.link(j) == i) {
                    return res;
                }
                // If the neighbor has deleted, then this connection should be repaired.
                if (isMarkedDeleted(cur.link(j))) {
                    res.connections_to_repair++;
                }
                inbound_connections_num[cur.link(j)]++;
                s.insert(cur.link(j));
                connections_checked++;

                // Check if this connection is bidirectional.
                LevelData &other = this->getLevelData(cur.link(j), l);
                for (int r = 0; r < other.numLinks(); r++) {
                    if (other.link(r) == (idType)i) {
                        double_connections++;
                        break;
                    }
                }
            }
            // Check if a certain neighbor appeared more than once.
            if (s.size() != cur.numLinks()) {
                return res;
            }
            incoming_edges_sets_sizes += cur.incomingEdges()->Get().size();
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
void HNSWIndex<DataType, DistType>::restoreGraph(std::ifstream &input) {	
	graphData.restore(input);
    for (idType id = 0; id < this->curElementCount; id++) {
		setVectorId(graphData.label(id), id);
	}
	
}

