#include <VecSim/vec_sim.h>

#include <utility>
#include "serialization.h"
#include "VecSim/utils/vecsim_stl.h"
#include "hnswlib.h"

#define HNSW_INVALID_META_DATA SIZE_MAX

namespace hnswlib {

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
    writeBinaryPOD(output, hnsw_index->max_id);
    writeBinaryPOD(output, hnsw_index->maxlevel_);
    writeBinaryPOD(output, hnsw_index->entrypoint_node_);

    // Save the available ids for reuse
    writeBinaryPOD(output, hnsw_index->available_ids.size());
    for (auto available_id : hnsw_index->available_ids) {
        writeBinaryPOD(output, available_id);
    }
}

void HNSWIndexSerializer::saveGraph(std::ofstream &output) {
    // Save level 0 data (graph layer 0 + labels + vectors data)
    output.write(hnsw_index->data_level0_memory_,
                 hnsw_index->max_elements_ * hnsw_index->size_data_per_element_);
    if (hnsw_index->max_id == HNSW_INVALID_ID) {
        return; // Index is empty.
    }
    // Save the incoming edge sets.
    for (size_t i = 0; i <= hnsw_index->max_id; i++) {
        if (hnsw_index->available_ids.find(i) != hnsw_index->available_ids.end()) {
            continue;
        }
        auto *set_ptr = hnsw_index->getIncomingEdgesPtr(i, 0);
        unsigned int set_size = set_ptr->size();
        writeBinaryPOD(output, set_size);
        for (auto id : *set_ptr) {
            writeBinaryPOD(output, id);
        }
    }

    // Save all graph layers other than layer 0: for every id of a vector in the graph,
    // store (<size>, data), where <size> is the data size, and the data is the concatenated
    // adjacency lists in the graph Then, store the sets of the incoming edges in every level.
    for (size_t i = 0; i <= hnsw_index->max_id; i++) {
        if (hnsw_index->available_ids.find(i) != hnsw_index->available_ids.end()) {
            continue;
        }
        unsigned int linkListSize =
            hnsw_index->element_levels_[i] > 0
                ? hnsw_index->size_links_per_element_ * hnsw_index->element_levels_[i]
                : 0;
        writeBinaryPOD(output, linkListSize);
        if (linkListSize)
            output.write(hnsw_index->linkLists_[i], linkListSize);
        for (size_t j = 1; j <= hnsw_index->element_levels_[i]; j++) {
            auto *set_ptr = hnsw_index->getIncomingEdgesPtr(i, j);
            unsigned int set_size = set_ptr->size();
            writeBinaryPOD(output, set_size);
            for (auto id : *set_ptr) {
                writeBinaryPOD(output, id);
            }
        }
    }
}

void HNSWIndexSerializer::restoreIndexFields(std::ifstream &input, SpaceInterface<float> *s) {
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

    if (s->get_data_size() != hnsw_index->data_size_) {
        throw std::runtime_error("Index data is corrupted or does not match the given space");
    }
    hnsw_index->fstdistfunc_ = s->get_dist_func();
    hnsw_index->dist_func_param_ = s->get_data_dim();

    // Restore index level generator of the top level for a new element
    readBinaryPOD(input, hnsw_index->level_generator_);

    // Restore index state
    readBinaryPOD(input, hnsw_index->cur_element_count);
    readBinaryPOD(input, hnsw_index->max_id);
    readBinaryPOD(input, hnsw_index->maxlevel_);
    readBinaryPOD(input, hnsw_index->entrypoint_node_);

    // Restore the available ids
    hnsw_index->available_ids.clear();
    size_t available_ids_count;
    readBinaryPOD(input, available_ids_count);
    for (size_t i = 0; i < available_ids_count; i++) {
        tableint next_id;
        readBinaryPOD(input, next_id);
        hnsw_index->available_ids.insert(next_id);
    }
}

void HNSWIndexSerializer::restoreGraph(std::ifstream &input) {
    // Restore graph layer 0
    hnsw_index->data_level0_memory_ = (char *)hnsw_index->allocator->reallocate(
        hnsw_index->data_level0_memory_,
        hnsw_index->max_elements_ * hnsw_index->size_data_per_element_);
    input.read(hnsw_index->data_level0_memory_,
               hnsw_index->max_elements_ * hnsw_index->size_data_per_element_);
    if (hnsw_index->max_id == HNSW_INVALID_ID) {
        return; // Index is empty.
    }
    for (size_t i = 0; i <= hnsw_index->max_id; i++) {
        if (hnsw_index->available_ids.find(i) != hnsw_index->available_ids.end()) {
            continue;
        }
        auto *set_ptr = new vecsim_stl::set<tableint>(hnsw_index->allocator);
        unsigned int set_size;
        readBinaryPOD(input, set_size);
        for (size_t j = 0; j < set_size; j++) {
            tableint next_edge;
            readBinaryPOD(input, next_edge);
            set_ptr->insert(next_edge);
        }
        hnsw_index->setIncomingEdgesPtr(i, 0, (void *)set_ptr);
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
    hnsw_index->label_lookup_.clear();

    for (size_t i = 0; i <= hnsw_index->max_id; i++) {
        if (hnsw_index->available_ids.find(i) != hnsw_index->available_ids.end()) {
            hnsw_index->element_levels_[i] = HNSW_INVALID_LEVEL;
            continue;
        }
        hnsw_index->label_lookup_[hnsw_index->getExternalLabel(i)] = i;
        unsigned int linkListSize;
        readBinaryPOD(input, linkListSize);
        if (linkListSize == 0) {
            hnsw_index->element_levels_[i] = 0;
            hnsw_index->linkLists_[i] = nullptr;
        } else {
            hnsw_index->element_levels_[i] = linkListSize / hnsw_index->size_links_per_element_;
            hnsw_index->linkLists_[i] = (char *)hnsw_index->allocator->allocate(linkListSize);
            if (hnsw_index->linkLists_[i] == nullptr)
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist");
            input.read(hnsw_index->linkLists_[i], linkListSize);
            for (size_t j = 1; j <= hnsw_index->element_levels_[i]; j++) {
                auto *set_ptr = new vecsim_stl::set<tableint>(hnsw_index->allocator);
                unsigned int set_size;
                readBinaryPOD(input, set_size);
                for (size_t k = 0; k < set_size; k++) {
                    tableint next_edge;
                    readBinaryPOD(input, next_edge);
                    set_ptr->insert(next_edge);
                }
                hnsw_index->setIncomingEdgesPtr(i, j, (void *)set_ptr);
            }
        }
    }
}

HNSWIndexSerializer::HNSWIndexSerializer(std::shared_ptr<HierarchicalNSW<float>> hnsw_index_) {
    hnsw_index = std::move(hnsw_index_); // Hold a reference of the index.
}

void HNSWIndexSerializer::saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    this->saveIndexFields(output);
    this->saveGraph(output);
    output.close();
}

void HNSWIndexSerializer::loadIndex(const std::string &location, SpaceInterface<float> *s) {

    std::ifstream input(location, std::ios::binary);
    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    input.seekg(0, std::ifstream::beg);
    this->restoreIndexFields(input, s);
    this->restoreGraph(input);
    input.close();
}

void HNSWIndexSerializer::reset(std::shared_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_) {
    // Hold the allocator, so it won't be destroyed while VecSimBaseObject is released.
    auto allocator = hnsw_index->getAllocator();
    if (hnsw_index_ == nullptr) {
        hnsw_index.reset();
        return;
    }
    hnsw_index = hnsw_index_;
}

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
    std::vector<int> inbound_connections_num(hnsw_index->max_id + 1, 0);
    size_t incoming_edges_sets_sizes = 0;

    for (size_t i = 0; i <= hnsw_index->max_id && hnsw_index->max_id != HNSW_INVALID_ID; i++) {
        if (hnsw_index->available_ids.find(i) != hnsw_index->available_ids.end()) {
            continue;
        }
        for (size_t l = 0; l <= hnsw_index->element_levels_[i]; l++) {
            linklistsizeint *ll_cur = hnsw_index->get_linklist_at_level(i, l);
            unsigned int size = hnsw_index->getListCount(ll_cur);
            auto *data = (tableint *)(ll_cur + 1);
            std::set<tableint> s;
            for (unsigned int j = 0; j < size; j++) {
                // Check if we found an invalid neighbor.
                if (data[j] > hnsw_index->max_id || data[j] == i) {
                    res.valid_state = false;
                    return res;
                }
                inbound_connections_num[data[j]]++;
                s.insert(data[j]);
                connections_checked++;

                // Check if this connection is bidirectional.
                linklistsizeint *ll_other = hnsw_index->get_linklist_at_level(data[j], l);
                int size_other = hnsw_index->getListCount(ll_other);
                auto *data_other = (tableint *)(ll_other + 1);
                for (int r = 0; r < size_other; r++) {
                    if (data_other[r] == (tableint)i) {
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
    std::cout << "HNSW index integrity OK\n";
    return res;
}
} // namespace hnswlib
