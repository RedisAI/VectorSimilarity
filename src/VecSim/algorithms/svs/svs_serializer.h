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

// Saves all reasonable fields of SVSIndex to the output stream
// This function saves all template parameters and instance fields needed to reconstruct
// an SVSIndex<MetricType, DataType, isMulti, QuantBits, ResidualBits, IsLeanVec>
void saveAllIndexFields(std::ofstream &output) const {
    // Save base class fields from VecSimIndexAbstract
    // Note: this->vecType corresponds to DataType template parameter
    // Note: this->metric corresponds to MetricType template parameter
    writeBinaryPOD(output, this->dim);
    writeBinaryPOD(output, this->vecType);      // DataType template parameter (as VecSimType enum)
    writeBinaryPOD(output, this->dataSize);
    writeBinaryPOD(output, this->metric);       // MetricType template parameter (as VecSimMetric enum)
    writeBinaryPOD(output, this->blockSize);
    writeBinaryPOD(output, this->isMulti);

    // Save SVS-specific configuration fields
    writeBinaryPOD(output, this->forcePreprocessing);
    writeBinaryPOD(output, this->changes_num);

    // Save build parameters
    writeBinaryPOD(output, this->buildParams.alpha);
    writeBinaryPOD(output, this->buildParams.graph_max_degree);
    writeBinaryPOD(output, this->buildParams.window_size);
    writeBinaryPOD(output, this->buildParams.max_candidate_pool_size);
    writeBinaryPOD(output, this->buildParams.prune_to);
    writeBinaryPOD(output, this->buildParams.use_full_search_history);

    // Save search parameters
    writeBinaryPOD(output, this->search_window_size);
    writeBinaryPOD(output, this->epsilon);

    // Save template parameters as metadata for validation during loading
    writeBinaryPOD(output, getCompressionMode());

    // QuantBits, ResidualBits, and IsLeanVec information

    // Save additional template parameter constants for complete reconstruction
    writeBinaryPOD(output, static_cast<size_t>(QuantBits));     // Template parameter QuantBits
    writeBinaryPOD(output, static_cast<size_t>(ResidualBits));  // Template parameter ResidualBits
    writeBinaryPOD(output, static_cast<bool>(IsLeanVec));       // Template parameter IsLeanVec
    writeBinaryPOD(output, static_cast<bool>(isMulti));         // Template parameter isMulti

    // Save additional metadata for validation during loading
    writeBinaryPOD(output, this->lastMode); // Last search mode
}

// Saves metadata (e.g., encoding version) to satisfy Serializer interface.
// Full index is saved separately in saveIndex() using file paths.
void saveIndexIMP(std::ofstream &output) override {
    EncodingVersion version = EncodingVersion_SVS_V0;
    writeBinaryPOD(output, version);

    // Save all index fields using the dedicated function
    saveAllIndexFields(output);
}

void saveIndex(const std::string &location) override {
    assert(impl_ && "Index is not initialized");
    if (impl_) {
        std::string verFile = location + "/metadata";
        std::ofstream output(verFile, std::ios::binary);
        saveIndexIMP(output);
        output.close();
        impl_->save(location + "/config", location + "/graph", location + "/data");
    }
}
