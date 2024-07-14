#pragma once

#include "abstract.hpp"
#include "data_block.h"

class BFSWithFiles : public BFS {
private:
    std::vector<DataBlock> blocks;
    size_t blockSize;

public:
    BFSWithFiles(size_t N, size_t maxLinks, size_t blockSize = 1024) : BFS(N, maxLinks), blocks(), blockSize(blockSize) {
        size_t numBlocks = N / blockSize + 1;
        blocks.reserve(numBlocks);
        char tmp[this->node_size];
        for (uint32_t i = 0; i < N; i++) {
            if (i % blockSize == 0) {
                blocks.emplace_back(blockSize, node_size);
            }
            node *cur = new (tmp) node();
            setNode(cur, i);
            blocks.back().addElement(cur);
        }
    }

    std::vector<nodeID_t> getNeighbors(nodeID_t id) const override {
        size_t blockIndex = id / blockSize;
        size_t indexInBlock = id % blockSize;
        auto data = blocks[blockIndex].getElement(indexInBlock);
        auto n = reinterpret_cast<const node *>(data);
        std::vector<nodeID_t> neighbors(n->links, n->links + n->numLinks);
        return neighbors;
    }
};
