/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include <fstream>
#include <string>

#include "VecSim/utils/serializer.h"

// Persist index into a file in the specified location.
void Serializer::saveIndex(const std::string &location) {

    // Serializing with the latest version.
    EncodingVersion version = EncodingVersion_V4;

    std::ofstream output(location, std::ios::binary);
    writeBinaryPOD(output, version);
    saveIndexIMP(output);
    output.close();
}

Serializer::EncodingVersion Serializer::ReadVersion(std::ifstream &input) {

    input.seekg(0, std::ifstream::beg);

    // The version number is the first field that is serialized.
    EncodingVersion version = EncodingVersion_HNSW_INVALID;
    readBinaryPOD(input, version);
    if (version <= EncodingVersion_DEPRECATED) {
        input.close();
        throw std::runtime_error("Cannot load index: deprecated encoding version: " +
                                 std::to_string(version));
    } else if (version >= EncodingVersion_HNSW_INVALID) {
        input.close();
        throw std::runtime_error("Cannot load index: bad encoding version: " +
                                 std::to_string(version));
    }
    return version;
}
