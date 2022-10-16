#pragma once

#include <ostream>
#include <istream>
// Helper functions for serializing the index.
template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *)&podRef, sizeof(T));
}

template <typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *)&podRef, sizeof(T));
}
typedef enum EncodingVersion { EncodingVersion_V1 = 1 } EncodingVersion;
