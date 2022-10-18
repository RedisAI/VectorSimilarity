#include <fstream>
#include <string>

#include "VecSim/utils/serializer.h"

// Persist index into a file in the specified location.
void Serializer::saveIndex(const std::string &location, EncodingVersion version) {
    std::ofstream output(location, std::ios::binary);
    output.write((char *)&version, sizeof(EncodingVersion));
    saveIndexIMP(output);
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
    // Only V1 is supported currently.
    if (version != EncodingVersion_V1) {
        throw std::runtime_error("Cannot load index: bad encoding version");
    }
    loadIndexIMP(input);
    input.close();

}
