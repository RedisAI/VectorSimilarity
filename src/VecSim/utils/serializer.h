
#pragma once

#include <ostream>
#include <istream>

class Serializer {
public:
    typedef enum EncodingVersion {
        EncodingVersion_V1 = 1,
        EncodingVersion_V2 = 2,
        EncodingVersion_INVALID, // This should always be last.
    } EncodingVersion;

    Serializer(EncodingVersion version = EncodingVersion_V2) : m_version(version) {}

    // Persist index into a file in the specified location with V2 encoding routine.
    void saveIndex(const std::string &location);

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
