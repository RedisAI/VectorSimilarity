#include <fstream>
#include <string>

#include "VecSim/utils/serializer.h"

// Persist index into a file in the specified location.
void Serializer::saveIndex(const std::string &location, EncodingVersion version) {
    // Only V1 and V2 are supported
    if (version >= EncodingVersion_NOT_VALID) {
        throw std::runtime_error("Cannot load index: bad encoding version");
    }

    std::ofstream output(location, std::ios::binary);
    writeBinaryPOD(output, version);
    saveIndexIMP(output, version);
    output.close();
}
// Restore the index from the file in the specified location.
void Serializer::loadIndex(const std::string &location) {
    std::ifstream input(location, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }
    input.seekg(0, std::ifstream::beg);

    // The version number is the first field that is serialized.
    EncodingVersion version;
    readBinaryPOD(input, version);
    // Only V1 and V2 are supported
    if (version >= EncodingVersion_NOT_VALID) {
        input.close();
        throw std::runtime_error("Cannot load index: bad encoding version");
    }

    loadIndexIMP(input, version);
    input.close();
}
