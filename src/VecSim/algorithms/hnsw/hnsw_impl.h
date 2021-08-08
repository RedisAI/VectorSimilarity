#include <queue>
#include <random>
#include <unordered_map>
#include "../../spaces/space_interface.h"
#include <set>
#include "visited_list_pool.h"

using namespace std;

namespace hnswlib {
typedef size_t labeltype;
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t> struct CompareByFirst {
    constexpr bool operator()(pair<dist_t, tableint> const &a,
                              pair<dist_t, tableint> const &b) const noexcept {
        return a.first < b.first;
    }
};

template <typename dist_t>
using CandidatesQueue = priority_queue<pair<dist_t, tableint>, std::vector<pair<dist_t, tableint>>,
                                       CompareByFirst<dist_t>>;

template <typename dist_t> class HierarchicalNSW {

    // Index build parameters
    size_t max_elements_;
    size_t M_;
    size_t maxM_;
    size_t maxM0_;
    size_t ef_construction_;

    // Index search parameter
    size_t ef_;

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
    size_t max_id;
    int maxlevel_;

    // Index data structures
    tableint enterpoint_node_;
    char *data_level0_memory_;
    char **linkLists_;
    std::vector<int> element_levels_;
    std::set<tableint> available_ids;
    std::unordered_map<labeltype, tableint> label_lookup_;
    VisitedListPool *visited_list_pool_;

    // used for synchronization only when parallel indexing / searching is enabled.
#ifdef ENABLE_PARALLELIZATION
    std::mutex global;
    std::mutex cur_element_count_guard_;
    std::vector<std::mutex> link_list_locks_;
#endif

    // callback for computing distance between two points in the underline space.
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_;

    labeltype getExternalLabel(tableint internal_id);
    void setExternalLabel(tableint internal_id, labeltype label);
    labeltype *getExternalLabelPtr(tableint internal_id);
    char *getDataByInternalId(tableint internal_id);
    int getRandomLevel(double reverse_size);
    std::set<tableint> *getIncomingEdgesPtr(tableint internal_id, int level);
    void setIncomingEdgesPtr(tableint internal_id, int level, void *set_ptr);
    linklistsizeint *get_linklist0(tableint internal_id);
    linklistsizeint *get_linklist(tableint internal_id, int level);
    linklistsizeint *get_linklist_at_level(tableint internal_id, int level);
    unsigned short int getListCount(const linklistsizeint *ptr);
    void setListCount(linklistsizeint *ptr, unsigned short int size);
    void removeExtraLinks(linklistsizeint *node_ll, CandidatesQueue<dist_t> candidates,
                          size_t Mcurmax, tableint *node_neighbors,
                          const std::set<tableint> &orig_neighbors, tableint *removed_links,
                          size_t *removed_links_num);
    CandidatesQueue<dist_t> searchLayer(tableint ep_id, const void *data_point, int layer,
                                        size_t ef);
    void getNeighborsByHeuristic2(CandidatesQueue<dist_t> &top_candidates, size_t M);
    tableint mutuallyConnectNewElement(tableint cur_c, CandidatesQueue<dist_t> &top_candidates,
                                       int level);
    void repairConnectionsForDeletion(tableint element_internal_id, tableint neighbour_id,
                                      tableint *neighbours_list,
                                      tableint *neighbour_neighbours_list, int level);

  public:
    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
                    size_t ef_construction = 200, size_t ef = 10, size_t random_seed = 100);
    ~HierarchicalNSW();

    void setEf(size_t ef);
    size_t getEf() const;
    size_t getIndexSize() const;
    size_t getIndexCapacity() const;
    size_t getEfConstruction() const;
    size_t getM() const;
    size_t getMaxLevel() const;
    void resizeIndex(size_t new_max_elements);
    bool removePoint(labeltype label);
    void addPoint(const void *data_point, labeltype label);
    priority_queue<pair<dist_t, labeltype>> searchKnn(const void *query_data, size_t k) const;
    void checkIntegrity();
};
} // namespace hnswlib
