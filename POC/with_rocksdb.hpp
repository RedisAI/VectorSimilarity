#pragma once

#include <stdio.h>

#include "abstract.hpp"
#include "rocksdb/include/rocksdb/c.h"

class BFSWithRocksDB : public BFS {
private:
    rocksdb_t *db;
    static constexpr char db_path[] = "/tmp/rocks.db";
    rocksdb_readoptions_t *readoptions;

public:
    BFSWithRocksDB(size_t N, size_t maxLinks) : BFS(N, maxLinks) {
        rocksdb_options_t *options = rocksdb_options_create();
        rocksdb_options_set_create_if_missing(options, 1);
        rename(db_path, "/tmp/rocks.db.bak"); // Verify a fresh start

        char *err = NULL;
        db = rocksdb_open(options, db_path, &err);

        rocksdb_writeoptions_t *writeoptions = rocksdb_writeoptions_create();
        readoptions = rocksdb_readoptions_create(); // for future use

        char tmpData[node_size];
        for (uint32_t i = 0; i < N; i++) {
            node *cur = new (tmpData) node();
            setNode(cur, i);
            rocksdb_put(db, writeoptions, reinterpret_cast<const char *>(&i), sizeof(i), tmpData, node_size, &err);
        }

        // cleanup
        rocksdb_writeoptions_destroy(writeoptions);
        rocksdb_options_destroy(options);
    }

    ~BFSWithRocksDB() { rocksdb_close(db); rocksdb_readoptions_destroy(readoptions); }

    std::vector<nodeID_t> getNeighbors(nodeID_t id) const override {
        size_t len;
        char *err;
        char *data = rocksdb_get(db, readoptions, reinterpret_cast<const char *>(&id), sizeof(id), &len, &err);
        auto n = reinterpret_cast<const node *>(data);
        std::vector<nodeID_t> neighbors(n->links, n->links + n->numLinks);
        free(data);
        return neighbors;
    }
};
