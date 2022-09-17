#pragma once

#include <cstddef>
#include <string>
#include "VecSim/algorithms/hnsw/hnsw.h"

// This struct is the return value of "checkIntegrity" methods (used for debugging).
typedef struct HNSWIndexMetaData {
    bool valid_state;
    long memory_usage; // in bytes
    size_t double_connections;
    size_t unidirectional_connections;
    size_t min_in_degree;
    size_t max_in_degree;
    size_t incoming_edges_mismatch;
} HNSWIndexMetaData;

typedef enum EncodingVersion { EncodingVersion_V1 = 1 } EncodingVersion;

class HNSWIndexSerializer {
private:
    // The serializer does not own the index, and it may be freed under the serializer feet.
    // Use the serializer with extra care, and dont use any function besides `reset` after
    // `VecSimIndex_Free` was invoked for the pointed index.
    HNSWIndex<float, float> *hnsw_index;

    void saveIndexFields(std::ofstream &output);
    void saveGraph(std::ofstream &output);
    void restoreIndexFields(std::ifstream &input);
    void restoreGraph(std::ifstream &input);
    void loadIndex_v1(std::ifstream &input);

public:
    // Wrap hnsw index.
    explicit HNSWIndexSerializer(HNSWIndex<float, float> *hnsw_index);

    // Persist HNSW index into a file in the specified location.
    void saveIndex(const std::string &location);

    // Check the validity of the reproduced index.
    HNSWIndexMetaData checkIntegrity();

    // Restore the index from the file in the specified location.
    void loadIndex(const std::string &location);

    // Safe release the inner hnsw_index pointer, optionally replace it with another.
    void reset(HNSWIndex<float, float> *hnsw_index = nullptr);
};
