/*
* Copyright (c) 2006-Present, Redis Ltd.
* All rights reserved.
*
* Licensed under your choice of the Redis Source Available License 2.0
* (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
* GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/multi.h"

// Saves metadata (e.g., encoding version) to satisfy Serializer interface.
// Full index is saved separately in saveIndex() using file paths.
void saveIndexIMP(std::ofstream &output) override {
    EncodingVersion version = EncodingVersion_SVS_V0;
    writeBinaryPOD(output, version);
}

void saveIndex(const std::string &location) override {
    assert(impl_ && "Index is not initialized");
    if (impl_) {
        std::string verFile = location + "metadata";
        std::ofstream output(verFile, std::ios::binary);
        saveIndexIMP(output);
        impl_->save(location + "/config", location + "/graph", location + "/data");

    }
}
