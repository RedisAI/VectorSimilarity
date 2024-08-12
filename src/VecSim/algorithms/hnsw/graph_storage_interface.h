#pragma once

#include <utility>

#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/vec_sim_common.h"

typedef uint16_t linkListSize;
using graphNodeType = std::pair<idType, unsigned short>; // represented as: (element_id, level)
using WriteBatch = void *;

class GraphStorageInterface {
    /**
     * Common struct to hold the outgoing edges of a node.
     */
    struct OutgoingEdges {
        linkListSize numLinks;
        idType links[];
    };
    /**
     * @brief Common struct to hold the incoming *unudirectional* edges of a node.
     */
    struct IncomingUnidirectionalEdges {
        vecsim_stl::vector<idType> incomingUnidirectionalEdges;
    };

public:
    GraphStorageInterface() = default;
    virtual ~GraphStorageInterface() = default;
    /**
     * @brief Initialize a new element in the graph storage.
     */
    virtual void intializeNewElement(idType node_id, size_t max_level) = 0;
    /**
     * @brief Add a new outgoing neighbor to the node.
     */
    virtual void addNeighbor(const graphNodeType &node, idType target, WriteBatch wb) = 0;
    /**
     * @brief Add a new incoming unidirectional neighbor to the node.
     */
    virtual std::vector<int> addIncomingUnidirectionalNeighbor(const graphNodeType &node,
                                                               idType nodeId, WriteBatch wb) = 0;
    /**
     * @brief update all edges for node (perform mutuall updates).
     */
    virtual void setAllNeighbors(const graphNodeType &node, std::vector<idType> nodeId,
                                 WriteBatch wb) = 0;
    /**
     * @brief get node's edges (read only).
     */
    virtual const OutgoingEdges &getEdges(const graphNodeType &node) = 0;
    /**
     * @brief get node's edges for update.
     */
    virtual OutgoingEdges &getEdgesForUpdate(const graphNodeType &node) = 0;
    /**
     * @brief get node's incoming unidirectional edges (read only).
     */
    virtual const IncomingUnidirectionalEdges &
    getIncomingUnidirectionalEdges(const graphNodeType &node) = 0;
    /**
     * @brief get node's incoming unidirectional edges for update.
     */
    virtual IncomingUnidirectionalEdges &
    getIncomingUnidirectionalEdgesForUpdate(const graphNodeType &node) = 0;
    /**
     * @brief mark element as permanently deleted from the graph storage when it has no longer
     * outgoing and incoming edges
     */
    virtual void permanentDelete(idType node_id) = 0;
    /**
     * @brief Apply garbage collection for permanente deleted elements from the graph storage.
     */
    virtual void applyGC() = 0;
    /**
     * Create new write trnasaction.
     */
    virtual WriteBatch createWriteBatch() = 0;
    /**
     * Commit write operaiton that were collected in the write batch.
     */
    virtual int commitWriteBatch(WriteBatch *wb) = 0;

    /**
     * @brief Save the graph storage to a file.
     */
    virtual void saveGraph(std::ofstream &output) const = 0;
    /**
     * @brief Load the graph storage from a file.
     */
    virtual void loadGraph(const std::ifstream &input) const = 0;
};
