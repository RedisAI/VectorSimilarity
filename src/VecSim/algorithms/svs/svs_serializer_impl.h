/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

#include "svs_serializer.h"
#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/multi.h"

// Saves all relevant fields of SVSIndex to the output stream
// This function saves all template parameters and instance fields needed to reconstruct
// an SVSIndex<MetricType, DataType, isMulti, QuantBits, ResidualBits, IsLeanVec>
template <typename MetricType, typename DataType, bool isMulti, size_t QuantBits,
          size_t ResidualBits, bool IsLeanVec>
void SVSIndex<MetricType, DataType, isMulti, QuantBits, ResidualBits, IsLeanVec>::saveIndexFields(
    std::ofstream &output) const {
    // Save base class fields from VecSimIndexAbstract
    // Note: this->vecType corresponds to DataType template parameter
    // Note: this->metric corresponds to MetricType template parameter
    writeBinaryPOD(output, this->dim);
    writeBinaryPOD(output, this->vecType); // DataType template parameter (as VecSimType enum)
    writeBinaryPOD(output, this->dataSize);
    writeBinaryPOD(output, this->metric); // MetricType template parameter (as VecSimMetric enum)
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
    writeBinaryPOD(output, static_cast<size_t>(QuantBits));    // Template parameter QuantBits
    writeBinaryPOD(output, static_cast<size_t>(ResidualBits)); // Template parameter ResidualBits
    writeBinaryPOD(output, static_cast<bool>(IsLeanVec));      // Template parameter IsLeanVec
    writeBinaryPOD(output, static_cast<bool>(isMulti));        // Template parameter isMulti

    // Save additional metadata for validation during loading
    writeBinaryPOD(output, this->lastMode); // Last search mode
}

// Saves metadata (e.g., encoding version) to satisfy Serializer interface.
// Full index is saved separately in saveIndex() using file paths.
template <typename MetricType, typename DataType, bool isMulti, size_t QuantBits,
          size_t ResidualBits, bool IsLeanVec>
void SVSIndex<MetricType, DataType, isMulti, QuantBits, ResidualBits, IsLeanVec>::saveIndexIMP(
    std::ofstream &output) {

    // Save all index fields using the dedicated function
    saveIndexFields(output);
}

// Saves metadata (e.g., encoding version) to satisfy Serializer interface.
// Full index is saved separately in saveIndex() using file paths.
template <typename MetricType, typename DataType, bool isMulti, size_t QuantBits,
          size_t ResidualBits, bool IsLeanVec>
void SVSIndex<MetricType, DataType, isMulti, QuantBits, ResidualBits, IsLeanVec>::impl_save(
    const std::string &location) {
    impl_->save(location + "/config", location + "/graph", location + "/data");
}

template <typename MetricType, typename DataType, bool isMulti, size_t QuantBits,
          size_t ResidualBits, bool IsLeanVec>
void SVSIndex<MetricType, DataType, isMulti, QuantBits, ResidualBits, IsLeanVec>::loadIndex(
    const std::string &folder_path) {
    svs::threads::ThreadPoolHandle threadpool_handle{VecSimSVSThreadPool{threadpool_}};
    // TODO rebase on master and use `logger_` field.
    // auto logger = makeLogger();

    if constexpr (isMulti) {
        auto loaded = svs::index::vamana::auto_multi_dynamic_assemble(
            folder_path + "/config",
            SVS_LAZY(graph_builder_t::load(folder_path + "/graph", this->blockSize,
                                           this->buildParams, this->getAllocator())),
            SVS_LAZY(storage_traits_t::load(folder_path + "/data", this->blockSize, this->dim,
                                            this->getAllocator())),
            distance_f(), std::move(threadpool_handle),
            svs::index::vamana::MultiMutableVamanaLoad::FROM_MULTI, logger_);
        impl_ = std::make_unique<impl_type>(std::move(loaded));
    } else {
        auto loaded = svs::index::vamana::auto_dynamic_assemble(
            folder_path + "/config",
            SVS_LAZY(graph_builder_t::load(folder_path + "/graph", this->blockSize,
                                           this->buildParams, this->getAllocator())),
            SVS_LAZY(storage_traits_t::load(folder_path + "/data", this->blockSize, this->dim,
                                            this->getAllocator())),
            distance_f(), std::move(threadpool_handle), false, logger_);
        impl_ = std::make_unique<impl_type>(std::move(loaded));
    }
}
