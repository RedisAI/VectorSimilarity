#pragma once

#include <stddef.h>
#include <random>
#include <vector>
#include <queue>
#include <set>

using nodeID_t = uint32_t;

struct node {
    nodeID_t numLinks;
    nodeID_t links[];
};

class BFS {
private:
    size_t maxLinks;
    size_t minLinks = 1;
    std::default_random_engine rand_gen;
    std::uniform_int_distribution<size_t> rand_dist;
protected:
    size_t N;
    size_t node_size;

public:
    BFS(size_t N, size_t maxLinks) : maxLinks(maxLinks), rand_dist(0, N - 1), N(N), node_size(sizeof(node) + sizeof(uint32_t) * maxLinks) {}
    ~BFS() = default;

    std::vector<nodeID_t> scanGraph(nodeID_t startNode = 0, size_t maxNodes = -1) const {
        size_t limit = std::min(maxNodes, N);
        std::vector<nodeID_t> nodes;
        std::set<nodeID_t> visited;
        std::queue<nodeID_t> candidates;
        candidates.push(startNode);
        while (!candidates.empty() && nodes.size() < limit) {
            nodeID_t cur = candidates.front();
            candidates.pop();
            if (visited.find(cur) != visited.end()) {
                continue;
            }
            visited.insert(cur);
            nodes.push_back(cur);
            std::vector<nodeID_t> neighbors = getNeighbors(cur);
            for (nodeID_t neighbor : neighbors) {
                candidates.push(neighbor);
            }
        }
        return nodes;
    }

    virtual std::vector<nodeID_t> getNeighbors(nodeID_t node) const = 0;

protected:
    void setNode(node *n, nodeID_t id) {
        n->numLinks = rand_dist(rand_gen) % (maxLinks - minLinks + 1) + minLinks;
        for (uint32_t j = 0; j < n->numLinks; j++) {
            n->links[j] = rand_dist(rand_gen);
            // Ensure no self loops or duplicate links (retry if necessary)
            if (n->links[j] == id) {
                j--;
                continue;
            }
            for (uint32_t k = 0; k < j; k++) {
                if (n->links[j] == n->links[k]) {
                    j--;
                    break;
                }
            }
        }
    }
};
