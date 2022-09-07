#pragma once

#include "visited_nodes_handler.h"
#include "VecSim/utils/arr_cpp.h"
#include "VecSim/memory/vecsim_malloc.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/utils/vec_utils.h"
#include "VecSim/query_result_struct.h"
#include "VecSim/vec_sim_common.h"
#include "VecSim/vec_sim_index.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h" //newBatchIterator

#include <random> //std::default_random_engine

typedef idType linklistsizeint;

template <typename DistType>
using candidatesMaxHeap = vecsim_stl::max_priority_queue<DistType, idType>;
template <typename DistType>
using candidatesLabelsMaxHeap = vecsim_stl::max_priority_queue<DistType, labelType>;

template <typename DataType, typename DistType>
class HNSWIndex : public VecSimIndexAbstract<DistType> {
private:
    // Index build parameters
    size_t max_elements_;
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;

    // Index search parameter
    size_t ef_;
    double epsilon_;

    // Index meta-data (based on the data dimensionality and index parameters)
    size_t data_size_;
    size_t size_data_per_element_;
    size_t size_links_per_element_;
    size_t size_links_level0_;
    size_t label_offset_;
    size_t offsetData_, offsetLevel0_;
    size_t incoming_links_offset0;
    size_t incoming_links_offset;
    double mult_;

    // Index level generator of the top level for a new element
    std::default_random_engine level_generator_;

    // Index state
    size_t cur_element_count;
    // TODO: after introducing the memory reclaim upon delete, max_id is redundant since the valid
    // internal ids are being kept as a continuous sequence [0, 1, ..,, cur_element_count-1].
    // We can remove this field completely if we change the serialization version, as the decoding
    // relies on this field.
    idType max_id;
    size_t maxlevel_;

    // Index data structures
    idType entrypoint_node_;
    char *data_level0_memory_;
    char **linkLists_;
    vecsim_stl::vector<size_t> element_levels_;
    vecsim_stl::unordered_map<labelType, idType> label_lookup_;
    std::shared_ptr<VisitedNodesHandler> visited_nodes_handler;

    // used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    std::unique_ptr<VisitedNodesHandlerPool> visited_nodes_handler_pool;
    size_t pool_initial_size;
    std::mutex global;
    std::mutex cur_element_count_guard_;
    std::vector<std::mutex> link_list_locks_;
#endif

#ifdef BUILD_TESTS
    friend class HNSWIndexSerializer;
    // Allow the following test to access the index size private member.
    friend class HNSWTest_preferAdHocOptimization_Test;
    friend class HNSWTest_test_dynamic_hnsw_info_iterator_Test;
    friend class AllocatorTest_testIncomingEdgesSet_Test;
    friend class AllocatorTest_test_hnsw_reclaim_memory_Test;
    friend class HNSWTest_testSizeEstimation_Test;
#endif

    HNSWIndex() = delete;                  // default constructor is disabled.
    HNSWIndex(const HNSWIndex &) = delete; // default (shallow) copy constructor is disabled.
    inline void setExternalLabel(idType internal_id, labelType label);
    inline labelType *getExternalLabelPtr(idType internal_id) const;
    inline size_t getRandomLevel(double reverse_size);
    inline vecsim_stl::vector<idType> *getIncomingEdgesPtr(idType internal_id, size_t level) const;
    inline void setIncomingEdgesPtr(idType internal_id, size_t level, void *edges_ptr);
    inline linklistsizeint *get_linklist0(idType internal_id) const;
    inline linklistsizeint *get_linklist(idType internal_id, size_t level) const;
    inline void setListCount(linklistsizeint *ptr, unsigned short int size);
    inline void removeExtraLinks(linklistsizeint *node_ll, candidatesMaxHeap<DistType> candidates,
                                 size_t Mcurmax, idType *node_neighbors,
                                 const vecsim_stl::vector<bool> &bitmap, idType *removed_links,
                                 size_t *removed_links_num);
    inline DistType processCandidate(idType curNodeId, const void *data_point, size_t layer,
                                     size_t ef, tag_t visited_tag,
                                     candidatesMaxHeap<DistType> &top_candidates,
                                     candidatesMaxHeap<DistType> &candidates_set,
                                     DistType lowerBound) const;
    inline void processCandidate_RangeSearch(idType curNodeId, const void *data_point, size_t layer,
                                             double epsilon, tag_t visited_tag,
                                             VecSimQueryResult **top_candidates,
                                             candidatesMaxHeap<DistType> &candidate_set,
                                             DistType lowerBound, DistType radius) const;
    candidatesMaxHeap<DistType> searchLayer(idType ep_id, const void *data_point, size_t layer,
                                            size_t ef) const;
    candidatesLabelsMaxHeap<DistType>
    searchBottomLayer_WithTimeout(idType ep_id, const void *data_point, size_t ef, size_t k,
                                  void *timeoutCtx, VecSimQueryResult_Code *rc) const;
    VecSimQueryResult *searchRangeBottomLayer_WithTimeout(idType ep_id, const void *data_point,
                                                          double epsilon, DistType radius,
                                                          void *timeoutCtx,
                                                          VecSimQueryResult_Code *rc) const;
    void getNeighborsByHeuristic2(candidatesMaxHeap<DistType> &top_candidates, size_t M);
    inline idType mutuallyConnectNewElement(idType cur_c,
                                            candidatesMaxHeap<DistType> &top_candidates,
                                            size_t level);
    void repairConnectionsForDeletion(idType element_internal_id, idType neighbour_id,
                                      idType *neighbours_list, idType *neighbour_neighbours_list,
                                      size_t level, vecsim_stl::vector<bool> &neighbours_bitmap);
    inline void replaceEntryPoint();
    inline void SwapLastIdWithDeletedId(idType element_internal_id,
                                        idType last_element_internal_id);

public:
    HNSWIndex(const HNSWParams *params, std::shared_ptr<VecSimAllocator> allocator,
              size_t random_seed = 100, size_t initial_pool_size = 1);
    virtual ~HNSWIndex();

    inline void setEf(size_t ef);
    inline size_t getEf() const;
    inline void setEpsilon(double epsilon);
    inline double getEpsilon() const;
    inline size_t indexSize() const override;
    inline size_t indexLabelCount() const override;
    inline size_t getIndexCapacity() const;
    inline size_t getEfConstruction() const;
    inline size_t getM() const;
    inline size_t getMaxLevel() const;
    inline size_t getEntryPointLabel() const;
    inline idType getEntryPointId() const;
    inline labelType getExternalLabel(idType internal_id) const;
    inline VisitedNodesHandler *getVisitedList() const;
    VecSimIndexInfo info() const override;
    VecSimInfoIterator *infoIterator() const override;
    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) override;
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) override;
    char *getDataByInternalId(idType internal_id) const;
    inline linklistsizeint *get_linklist_at_level(idType internal_id, size_t level) const;
    inline unsigned short int getListCount(const linklistsizeint *ptr) const;
    inline void resizeIndex(size_t new_max_elements);
    int deleteVector(labelType label) override;
    int addVector(const void *vector_data, labelType label) override;
    double getDistanceFrom(labelType label, const void *vector_data) const override;
    inline idType searchBottomLayerEP(const void *query_data, void *timeoutCtx,
                                      VecSimQueryResult_Code *rc) const;
    VecSimQueryResult_List topKQuery(const void *query_data, size_t k,
                                     VecSimQueryParams *queryParams) override;
    VecSimQueryResult_List rangeQuery(const void *query_data, DistType radius,
                                      VecSimQueryParams *queryParams) override;
};
