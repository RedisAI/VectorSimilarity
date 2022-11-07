#include <VecSim/vec_sim.h>

#include <utility>
#include "serialization.h"
#include "VecSim/utils/vecsim_stl.h"
#include "hnsw_single.h"
#define HNSW_INVALID_META_DATA SIZE_MAX

// Helper functions for serializing the index.
template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *)&podRef, sizeof(T));
}

void HNSWIndexSerializer::saveIndexFields(std::ofstream &output) {
    // Save index build parameters
    writeBinaryPOD(output, hnsw_index->max_elements_);
    writeBinaryPOD(output, hnsw_index->M_);
    writeBinaryPOD(output, hnsw_index->maxM_);
    writeBinaryPOD(output, hnsw_index->maxM0_);
    writeBinaryPOD(output, hnsw_index->ef_construction_);

    // Save index search parameter
    writeBinaryPOD(output, hnsw_index->ef_);

    // Save index meta-data
    writeBinaryPOD(output, hnsw_index->data_size_);
    writeBinaryPOD(output, hnsw_index->size_data_per_element_);
    writeBinaryPOD(output, hnsw_index->size_links_per_element_);
    writeBinaryPOD(output, hnsw_index->size_links_level0_);
    writeBinaryPOD(output, hnsw_index->label_offset_);
    writeBinaryPOD(output, hnsw_index->offsetData_);
    writeBinaryPOD(output, hnsw_index->offsetLevel0_);
    writeBinaryPOD(output, hnsw_index->incoming_links_offset0);
    writeBinaryPOD(output, hnsw_index->incoming_links_offset);
    writeBinaryPOD(output, hnsw_index->mult_);

    // Save index level generator of the top level for a new element
    writeBinaryPOD(output, hnsw_index->level_generator_);

    // Save index state
    writeBinaryPOD(output, hnsw_index->cur_element_count);
    writeBinaryPOD(output, hnsw_index->num_marked_deleted);
    writeBinaryPOD(output, hnsw_index->maxlevel_);
    writeBinaryPOD(output, hnsw_index->entrypoint_node_);
}

void HNSWIndexSerializer::saveGraph(std::ofstream &output) {
    // Save level 0 data (graph layer 0 + labels + vectors data)
    output.write(hnsw_index->data_level0_memory_,
                 hnsw_index->max_elements_ * hnsw_index->size_data_per_element_);

    // Save the incoming edge sets.
    for (size_t i = 0; i < hnsw_index->cur_element_count; i++) {
        auto *incoming_edges_ptr = hnsw_index->getIncomingEdgesPtr(i, 0);
        unsigned int set_size = incoming_edges_ptr->size();
        writeBinaryPOD(output, set_size);
        for (auto id : *incoming_edges_ptr) {
            writeBinaryPOD(output, id);
        }
    }

    // Save all graph layers other than layer 0: for every id of a vector in the graph,
    // store (<size>, data), where <size> is the data size, and the data is the concatenated
    // adjacency lists in the graph Then, store the sets of the incoming edges in every level.
    for (size_t i = 0; i < hnsw_index->cur_element_count; i++) {
        unsigned int llSize =
            hnsw_index->element_levels_[i] > 0
                ? hnsw_index->size_links_per_element_ * hnsw_index->element_levels_[i]
                : 0;
        writeBinaryPOD(output, llSize);
        if (llSize)
            output.write(hnsw_index->linkLists_[i], llSize);
        for (size_t j = 1; j <= hnsw_index->element_levels_[i]; j++) {
            auto *incoming_edges_ptr = hnsw_index->getIncomingEdgesPtr(i, j);
            unsigned int set_size = incoming_edges_ptr->size();
            writeBinaryPOD(output, set_size);
            for (auto id : *incoming_edges_ptr) {
                writeBinaryPOD(output, id);
            }
        }
    }
}

void HNSWIndexSerializer::loadIndex_v1(std::ifstream &input) {
    // Restore index build parameters
    readBinaryPOD(input, hnsw_index->max_elements_);
    readBinaryPOD(input, hnsw_index->M_);
    readBinaryPOD(input, hnsw_index->maxM_);
    readBinaryPOD(input, hnsw_index->maxM0_);
    readBinaryPOD(input, hnsw_index->ef_construction_);

    // Restore index search parameter
    readBinaryPOD(input, hnsw_index->ef_);

    // Restore index meta-data
    readBinaryPOD(input, hnsw_index->data_size_);
    readBinaryPOD(input, hnsw_index->size_data_per_element_);
    readBinaryPOD(input, hnsw_index->size_links_per_element_);
    readBinaryPOD(input, hnsw_index->size_links_level0_);
    readBinaryPOD(input, hnsw_index->label_offset_);
    readBinaryPOD(input, hnsw_index->offsetData_);
    readBinaryPOD(input, hnsw_index->offsetLevel0_);
    readBinaryPOD(input, hnsw_index->incoming_links_offset0);
    readBinaryPOD(input, hnsw_index->incoming_links_offset);
    readBinaryPOD(input, hnsw_index->mult_);

    // Restore index level generator of the top level for a new element
    readBinaryPOD(input, hnsw_index->level_generator_);

    // Restore index state
    readBinaryPOD(input, hnsw_index->cur_element_count);
    idType dummy;
    readBinaryPOD(input, dummy);
    hnsw_index->num_marked_deleted = 0;
    readBinaryPOD(input, hnsw_index->maxlevel_);
    readBinaryPOD(input, hnsw_index->entrypoint_node_);

    this->restoreGraph(input);
    size_t old_size_links_per_element_ = hnsw_index->size_links_per_element_;
    hnsw_index->size_links_per_element_ -= sizeof(elementFlags);
    hnsw_index->incoming_links_offset -= sizeof(elementFlags);
    for (idType i = 0; i < hnsw_index->cur_element_count; i++) {
        auto meta = hnsw_index->get_linklist0(i);
        *(meta) = *(meta - 1);
        *(meta - 1) = 0;
        size_t llSize = hnsw_index->element_levels_[i] * hnsw_index->size_links_per_element_;
        if (llSize) {
            char *levels_data = (char *)hnsw_index->allocator->allocate(llSize);
            for (size_t offset = 0; offset < hnsw_index->element_levels_[i]; offset++) {
                memcpy(levels_data + offset * hnsw_index->size_links_per_element_ + 2,
                       hnsw_index->linkLists_[i] + offset * old_size_links_per_element_ + 4,
                       hnsw_index->size_links_per_element_ - 2);
                *(linkListSize *)(levels_data + offset * hnsw_index->size_links_per_element_) =
                    *(idType *)(hnsw_index->linkLists_[i] + offset * old_size_links_per_element_);
            }
            hnsw_index->allocator->free_allocation(hnsw_index->linkLists_[i]);
            hnsw_index->linkLists_[i] = levels_data;
        }
    }
}

void HNSWIndexSerializer::loadIndex_v2(std::ifstream &input) {
    this->restoreIndexFields(input);
    this->restoreGraph(input);
}

void HNSWIndexSerializer::restoreIndexFields(std::ifstream &input) {
    // Restore index build parameters
    readBinaryPOD(input, hnsw_index->max_elements_);
    readBinaryPOD(input, hnsw_index->M_);
    readBinaryPOD(input, hnsw_index->maxM_);
    readBinaryPOD(input, hnsw_index->maxM0_);
    readBinaryPOD(input, hnsw_index->ef_construction_);

    // Restore index search parameter
    readBinaryPOD(input, hnsw_index->ef_);

    // Restore index meta-data
    readBinaryPOD(input, hnsw_index->data_size_);
    readBinaryPOD(input, hnsw_index->size_data_per_element_);
    readBinaryPOD(input, hnsw_index->size_links_per_element_);
    readBinaryPOD(input, hnsw_index->size_links_level0_);
    readBinaryPOD(input, hnsw_index->label_offset_);
    readBinaryPOD(input, hnsw_index->offsetData_);
    readBinaryPOD(input, hnsw_index->offsetLevel0_);
    readBinaryPOD(input, hnsw_index->incoming_links_offset0);
    readBinaryPOD(input, hnsw_index->incoming_links_offset);
    readBinaryPOD(input, hnsw_index->mult_);

    // Restore index level generator of the top level for a new element
    readBinaryPOD(input, hnsw_index->level_generator_);

    // Restore index state
    readBinaryPOD(input, hnsw_index->cur_element_count);
    readBinaryPOD(input, hnsw_index->num_marked_deleted);
    readBinaryPOD(input, hnsw_index->maxlevel_);
    readBinaryPOD(input, hnsw_index->entrypoint_node_);
}

void HNSWIndexSerializer::restoreGraph(std::ifstream &input) {
    // Restore graph layer 0
    hnsw_index->data_level0_memory_ = (char *)hnsw_index->allocator->reallocate(
        hnsw_index->data_level0_memory_,
        hnsw_index->max_elements_ * hnsw_index->size_data_per_element_);
    input.read(hnsw_index->data_level0_memory_,
               hnsw_index->max_elements_ * hnsw_index->size_data_per_element_);
    if (hnsw_index->cur_element_count == 0) {
        return; // Index is empty.
    }
    for (size_t i = 0; i < hnsw_index->cur_element_count; i++) {
        auto *incoming_edges =
            new (hnsw_index->allocator) vecsim_stl::vector<idType>(hnsw_index->allocator);
        unsigned int incoming_edges_len;
        readBinaryPOD(input, incoming_edges_len);
        for (size_t j = 0; j < incoming_edges_len; j++) {
            idType next_edge;
            readBinaryPOD(input, next_edge);
            incoming_edges->push_back(next_edge);
        }
        hnsw_index->setIncomingEdgesPtr(i, 0, (void *)incoming_edges);
    }

#ifdef ENABLE_PARALLELIZATION
    pool_initial_size = pool_initial_size;
    visited_nodes_handler_pool = std::unique_ptr<VisitedNodesHandlerPool>(
        new (this->allocator)
            VisitedNodesHandlerPool(pool_initial_size, max_elements_, this->allocator));
#else
    hnsw_index->visited_nodes_handler = std::unique_ptr<VisitedNodesHandler>(
        new (hnsw_index->allocator)
            VisitedNodesHandler(hnsw_index->max_elements_, hnsw_index->allocator));
#endif

    // Restore the rest of the graph layers, along with the label and max_level lookups.
    hnsw_index->linkLists_ = (char **)hnsw_index->allocator->reallocate(
        hnsw_index->linkLists_, sizeof(void *) * hnsw_index->max_elements_);
    hnsw_index->element_levels_ =
        vecsim_stl::vector<size_t>(hnsw_index->max_elements_, hnsw_index->allocator);
    reinterpret_cast<HNSWIndex_Single<float, float> *>(hnsw_index)->label_lookup_.clear();

    for (size_t i = 0; i < hnsw_index->cur_element_count; i++) {
        reinterpret_cast<HNSWIndex_Single<float, float> *>(hnsw_index)
            ->label_lookup_[hnsw_index->getExternalLabel(i)] = i;
        unsigned int llSize;
        readBinaryPOD(input, llSize);
        if (llSize == 0) {
            hnsw_index->element_levels_[i] = 0;
            hnsw_index->linkLists_[i] = nullptr;
        } else {
            hnsw_index->element_levels_[i] = llSize / hnsw_index->size_links_per_element_;
            hnsw_index->linkLists_[i] = (char *)hnsw_index->allocator->allocate(llSize);
            if (hnsw_index->linkLists_[i] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist");
            input.read(hnsw_index->linkLists_[i], llSize);
            for (size_t j = 1; j <= hnsw_index->element_levels_[i]; j++) {
                auto *incoming_edges =
                    new (hnsw_index->allocator) vecsim_stl::vector<idType>(hnsw_index->allocator);
                unsigned int vector_len;
                readBinaryPOD(input, vector_len);
                for (size_t k = 0; k < vector_len; k++) {
                    idType next_edge;
                    readBinaryPOD(input, next_edge);
                    incoming_edges->push_back(next_edge);
                }
                hnsw_index->setIncomingEdgesPtr(i, j, (void *)incoming_edges);
            }
        }
    }
}

HNSWIndexSerializer::HNSWIndexSerializer(HNSWIndex<float, float> *hnsw_index_)
    : hnsw_index(hnsw_index_) {}

void HNSWIndexSerializer::saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    EncodingVersion version = EncodingVersion_V2;
    output.write((char *)&version, sizeof(EncodingVersion));
    this->saveIndexFields(output);
    this->saveGraph(output);
    output.close();
}

void HNSWIndexSerializer::loadIndex(const std::string &location) {

    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }
    input.seekg(0, std::ifstream::beg);

    // The version number is the first field that is serialized.
    EncodingVersion version;
    readBinaryPOD(input, version);
    switch (version) {
    case EncodingVersion_V2:
        loadIndex_v2(input);
        break;

    case EncodingVersion_V1:
        loadIndex_v1(input);
        break;

    default:
        throw std::runtime_error("Cannot load index: bad encoding version");
    }
    input.close();
}

// The serializer does not own the index, here we just replace the pointed index.
void HNSWIndexSerializer::reset(HNSWIndex<float, float> *hnsw_index_) { hnsw_index = hnsw_index_; }

HNSWIndexMetaData HNSWIndexSerializer::checkIntegrity() {
    HNSWIndexMetaData res = {.valid_state = false,
                             .memory_usage = -1,
                             .double_connections = HNSW_INVALID_META_DATA,
                             .unidirectional_connections = HNSW_INVALID_META_DATA,
                             .min_in_degree = HNSW_INVALID_META_DATA,
                             .max_in_degree = HNSW_INVALID_META_DATA};

    // Save the current memory usage (before we use additional memory for the integrity check).
    res.memory_usage = hnsw_index->getAllocator()->getAllocationSize();
    size_t connections_checked = 0, double_connections = 0;
    std::vector<int> inbound_connections_num(hnsw_index->cur_element_count, 0);
    size_t incoming_edges_sets_sizes = 0;
    for (size_t i = 0; i < hnsw_index->cur_element_count; i++) {
        for (size_t l = 0; l <= hnsw_index->element_levels_[i]; l++) {
            linkListSize *ll_cur = hnsw_index->get_linklist_at_level(i, l);
            unsigned short size = hnsw_index->getListCount(ll_cur);
            auto *data = (idType *)(ll_cur + 1);
            std::set<idType> s;
            for (unsigned short j = 0; j < size; j++) {
                // Check if we found an invalid neighbor.
                if (data[j] >= hnsw_index->cur_element_count || data[j] == i) {
                    res.valid_state = false;
                    return res;
                }
                inbound_connections_num[data[j]]++;
                s.insert(data[j]);
                connections_checked++;

                // Check if this connection is bidirectional.
                linkListSize *ll_other = hnsw_index->get_linklist_at_level(data[j], l);
                unsigned short size_other = hnsw_index->getListCount(ll_other);
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
            incoming_edges_sets_sizes += hnsw_index->getIncomingEdgesPtr(i, l)->size();
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
