/*
* Copyright (c) 2006-Present, Redis Ltd.
* All rights reserved.
*
* Licensed under your choice of the Redis Source Available License 2.0
* (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
* GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include <fstream>
#include <string>
#include "VecSim/utils/serializer.h"

class HNSWserializer : public Serializer {
    public:
    enum class EncodingVersion {
        DEPRECATED = 2, // Last deprecated version
        V3,
        V4,
        INVALID
    };

    HNSWserializer(EncodingVersion version = EncodingVersion::V4) : m_version(version) {};
    static EncodingVersion ReadVersion(std::ifstream &input) {

        input.seekg(0, std::ifstream::beg);

        // The version number is the first field that is serialized.
        EncodingVersion version = EncodingVersion::INVALID;
        readBinaryPOD(input, version);

        if (version <= EncodingVersion::DEPRECATED) {
            input.close();
            throw std::runtime_error("Cannot load index: deprecated encoding version: " +
                                     std::to_string(static_cast<int>(version)));
        } else if (version >= EncodingVersion::INVALID) {
            input.close();
            throw std::runtime_error("Cannot load index: bad encoding version: " +
                                    std::to_string(static_cast<int>(version)));
        }
        return version;
    }

    // Persist index into a file in the specified location.
    void saveIndex(const std::string &location) {

        // Serializing with the latest version.
        // Using int to enable multiple EncodingVersions types
        EncodingVersion version = EncodingVersion::V4;
        std::ofstream output(location, std::ios::binary);
        writeBinaryPOD(output, version);
        saveIndexIMP(output);
        output.close();
    }

    EncodingVersion getVersion() const  {
        return m_version;
    }

    protected:
    EncodingVersion m_version;
};
