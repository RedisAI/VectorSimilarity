#include <fstream>
#include <string>

#include "VecSim/utils/serializer.h"

// Persist index into a file in the specified location.
void Serializer::saveIndex(const std::string &location) {

    // Serializing with V2.
    EncodingVersion version = EncodingVersion_V2;

    std::ofstream output(location, std::ios::binary);
    writeBinaryPOD(output, version);
    saveIndexIMP(output);
    output.close();
}

Serializer::EncodingVersion Serializer::ReadVersion(std::ifstream &input) {

    input.seekg(0, std::ifstream::beg);

    // The version number is the first field that is serialized.
    EncodingVersion version = EncodingVersion_INVALID;
    readBinaryPOD(input, version);
    // Only V1 and V2 are supported
    if (version <= 0 || version >= EncodingVersion_INVALID) {
        input.close();
        throw std::runtime_error("Cannot load index: bad encoding version");
    }

    return version;
}
