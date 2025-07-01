/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once

#include <ostream>
#include <istream>

class Serializer {
public:
    typedef enum EncodingVersion {
        // HNSW Enconding versions
        EncodingVersion_DEPRECATED = 2, // Last deprecated version
        EncodingVersion_V3,
        EncodingVersion_V4,
        // SVS Enconding versions
        EncodingVersion_SVS_V0 = 1000, // Do not change this value
        EncodingVersion_INVALID, // This should always be last.
    } EncodingVersion;

    Serializer(EncodingVersion version = EncodingVersion_V4) : m_version(version) {}

    // Persist index into a file in the specified location with V3 encoding routine.
    virtual void saveIndex(const std::string &location);

    EncodingVersion getVersion() const { return m_version; }

    static EncodingVersion ReadVersion(std::ifstream &input);

    // Helper functions for serializing the index.
    template <typename T>
    static inline void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *)&podRef, sizeof(T));
    }

    template <typename T>
    static inline void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *)&podRef, sizeof(T));
    }

protected:
    EncodingVersion m_version;

    // Index memory size might be changed during index saving.
    virtual void saveIndexIMP(std::ofstream &output) = 0;
};
