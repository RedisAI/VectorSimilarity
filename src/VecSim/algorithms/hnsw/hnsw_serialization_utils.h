#pragma once

#include <cstddef>

#define HNSW_INVALID_META_DATA SIZE_MAX

typedef struct {
    bool valid_state;
    long memory_usage; // in bytes
    size_t double_connections;
    size_t unidirectional_connections;
    size_t min_in_degree;
    size_t max_in_degree;
    size_t connections_to_repair;
} HNSWIndexMetaData;
