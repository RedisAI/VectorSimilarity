/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#pragma once
#include "VecSim/vec_sim_index.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/vec_utils.h"

#include <cmath>
#include <memory>
#include <cassert>
#include <limits>
#include <vector>

#include "svs/index/vamana/dynamic_index.h"
#include "svs/index/vamana/multi.h"
#include "spdlog/sinks/callback_sink.h"

#include "VecSim/algorithms/svs/svs_utils.h"
#include "VecSim/algorithms/svs/svs_batch_iterator.h"
#include "VecSim/algorithms/svs/svs_extensions.h"

#include "svs_serializer.h"

struct SVSIndexBase
#ifdef BUILD_TESTS
    : public SVSserializer
#endif
{

    virtual ~SVSIndexBase() = default;
    virtual int addVectors(const void *vectors_data, const labelType *labels, size_t n) = 0;
    virtual int deleteVectors(const labelType *labels, size_t n) = 0;
    virtual size_t indexStorageSize() const = 0;
    virtual size_t getNumThreads() const = 0;
    virtual void setNumThreads(size_t numThreads) = 0;
    virtual size_t getThreadPoolCapacity() const = 0;
    virtual bool isCompressed() const = 0;
    virtual void loadIndex(const std::string &folder_path)  { return;};
#ifdef BUILD_TESTS
    virtual svs::logging::logger_ptr getLogger() const = 0;
#endif
};

template <typename MetricType, typename DataType, bool isMulti, size_t QuantBits,
          size_t ResidualBits, bool IsLeanVec>
class SVSIndex : public VecSimIndexAbstract<svs_details::vecsim_dt<DataType>, float>,
                 public SVSIndexBase {
protected:
    using data_type = DataType;
    using distance_f = MetricType;
    using Base = VecSimIndexAbstract<svs_details::vecsim_dt<DataType>, float>;
    using index_component_t = IndexComponents<svs_details::vecsim_dt<DataType>, float>;

    using storage_traits_t = SVSStorageTraits<DataType, QuantBits, ResidualBits, IsLeanVec>;
    using index_storage_type = typename storage_traits_t::index_storage_type;

    using graph_builder_t = SVSGraphBuilder<uint32_t>;
    using graph_type = typename graph_builder_t::graph_type;

    using impl_type = std::conditional_t<
        isMulti,
        svs::index::vamana::MultiMutableVamanaIndex<graph_type, index_storage_type, distance_f>,
        svs::index::vamana::MutableVamanaIndex<graph_type, index_storage_type, distance_f>>;

    bool forcePreprocessing;

    // Index severe changes counter to initiate reindexing if number of changes exceed threshold
    // markIndexUpdated() manages this counter
    size_t changes_num;

    // Index build parameters
    svs::index::vamana::VamanaBuildParameters buildParams;

    // Index search parameters
    size_t search_window_size;
    size_t search_buffer_capacity;
    // LeanVec dataset dimension
    // This parameter allows to tune LeanVec dimension if LeanVec is enabled
    size_t leanvec_dim;
    double epsilon;

    // Check if the dataset is Two-level LVQ
    // This allows to tune default window capacity during search
    bool is_two_level_lvq;

    // SVS thread pool
    VecSimSVSThreadPool threadpool_;
    svs::logging::logger_ptr logger_;
    // SVS Index implementation instance
    std::unique_ptr<impl_type> impl_;

    static double toVecSimDistance(float v) { return svs_details::toVecSimDistance<distance_f>(v); }

    svs::logging::logger_ptr makeLogger() {
        spdlog::custom_log_callback callback = [this](const spdlog::details::log_msg &msg) {
            if (!VecSimIndexInterface::logCallback) {
                return; // No callback function provided
            }
            // Custom callback implementation
            const char *vecsim_level = [msg]() {
                switch (msg.level) {
                case spdlog::level::trace:
                    return VecSimCommonStrings::LOG_DEBUG_STRING;
                case spdlog::level::debug:
                    return VecSimCommonStrings::LOG_VERBOSE_STRING;
                case spdlog::level::info:
                    return VecSimCommonStrings::LOG_NOTICE_STRING;
                case spdlog::level::warn:
                case spdlog::level::err:
                case spdlog::level::critical:
                    return VecSimCommonStrings::LOG_WARNING_STRING;
                default:
                    return "UNKNOWN";
                }
            }();

            std::string msg_str{msg.payload.data(), msg.payload.size()};
            // Log the message using the custom callback
            VecSimIndexInterface::logCallback(this->logCallbackCtx, vecsim_level, msg_str.c_str());
        };

        // Create a logger with the custom callback
        auto sink = std::make_shared<spdlog::sinks::callback_sink_mt>(callback);
        auto logger = std::make_shared<spdlog::logger>("SVSIndex", sink);
        // Sink all messages to VecSim
        logger->set_level(spdlog::level::trace);
        return logger;
    }

    // Create SVS index instance with initial data
    // Data should not be empty
    template <svs::data::ImmutableMemoryDataset Dataset>
    void initImpl(const Dataset &points, std::span<const labelType> ids) {
        svs::threads::ThreadPoolHandle threadpool_handle{VecSimSVSThreadPool{threadpool_}};

        // Construct SVS index initial storage with compression if needed
        auto data = storage_traits_t::create_storage(points, this->blockSize, threadpool_handle,
                                                     this->getAllocator(), this->leanvec_dim);
        // Compute the entry point.
        auto entry_point =
            svs::index::vamana::extensions::compute_entry_point(data, threadpool_handle);

        // Perform graph construction.
        auto distance = distance_f{};
        const auto &parameters = this->buildParams;

        // Construct initial Vamana Graph
        auto graph =
            graph_builder_t::build_graph(parameters, data, distance, threadpool_, entry_point,
                                         this->blockSize, this->getAllocator(), logger_);

        // Create SVS MutableIndex instance
        impl_ = std::make_unique<impl_type>(std::move(graph), std::move(data), entry_point,
                                            std::move(distance), ids, threadpool_, logger_);

        // Set SVS MutableIndex build parameters to be used in future updates
        impl_->set_construction_window_size(parameters.window_size);
        impl_->set_max_candidates(parameters.max_candidate_pool_size);
        impl_->set_prune_to(parameters.prune_to);
        impl_->set_alpha(parameters.alpha);
        impl_->set_full_search_history(parameters.use_full_search_history);

        // Configure default search parameters
        auto sp = impl_->get_search_parameters();
        sp.buffer_config({this->search_window_size, this->search_buffer_capacity});
        impl_->set_search_parameters(sp);
        impl_->reset_performance_parameters();
    }

    // Preprocess batch of vectors
    MemoryUtils::unique_blob preprocessForBatchStorage(const void *original_data, size_t n) const {
        // Buffer alignment isn't necessary for storage since SVS index will copy the data
        if (!this->forcePreprocessing) {
            return MemoryUtils::unique_blob{const_cast<void *>(original_data), [](void *) {}};
        }

        const auto data_size = this->getDataSize() * n;

        auto processed_blob =
            MemoryUtils::unique_blob{this->allocator->allocate(data_size),
                                     [this](void *ptr) { this->allocator->free_allocation(ptr); }};
        // Assuming original data size equals to processed data size
        memcpy(processed_blob.get(), original_data, data_size);
        // Preprocess each vector in place
        for (size_t i = 0; i < n; i++) {
            this->preprocessStorageInPlace(static_cast<DataType *>(processed_blob.get()) +
                                           i * this->dim);
        }
        return processed_blob;
    }

    int addVectorsImpl(const void *vectors_data, const labelType *labels, size_t n) {
        if (n == 0) {
            return 0;
        }

        int deleted_num = 0;
        if constexpr (!isMulti) {
            // SVS index does not support overriding vectors with the same label
            // so we have to delete them first if needed
            deleted_num = deleteVectorsImpl(labels, n);
        }

        std::span<const labelType> ids(labels, n);
        auto processed_blob = this->preprocessForBatchStorage(vectors_data, n);
        auto typed_vectors_data = static_cast<DataType *>(processed_blob.get());
        // Wrap data into SVS SimpleDataView for SVS API
        auto points = svs::data::SimpleDataView<DataType>{typed_vectors_data, n, this->dim};

        // If n == 1, we should ensure single-threading
        const size_t current_num_threads = getNumThreads();
        if (n == 1 && current_num_threads > 1) {
            setNumThreads(1);
        }

        if (!impl_) {
            // SVS index instance cannot be empty, so we have to construct it at first rows
            initImpl(points, ids);
        } else {
            // Add new points to existing SVS index
            impl_->add_points(points, ids);
        }

        // Restore multi-threading if needed
        if (n == 1 && current_num_threads > 1) {
            setNumThreads(current_num_threads);
        }

        return n - deleted_num;
    }

    int deleteVectorsImpl(const labelType *labels, size_t n) {
        if (indexSize() == 0) {
            return 0;
        }

        // SVS fails if we try to delete non-existing entries
        std::vector<labelType> entries_to_delete;
        entries_to_delete.reserve(n);
        for (size_t i = 0; i < n; i++) {
            if (impl_->has_id(labels[i])) {
                entries_to_delete.push_back(labels[i]);
            }
        }

        if (entries_to_delete.size() == 0) {
            return 0;
        }

        // If entries_to_delete.size() == 1, we should ensure single-threading
        const size_t current_num_threads = getNumThreads();
        if (n == 1 && current_num_threads > 1) {
            setNumThreads(1);
        }

        const auto deleted_num = impl_->delete_entries(entries_to_delete);

        // Restore multi-threading if needed
        if (n == 1 && current_num_threads > 1) {
            setNumThreads(current_num_threads);
        }

        this->markIndexUpdate(deleted_num);
        return deleted_num;
    }

    // Count severe index changes (currently deletions only) and consolidate index if needed
    void markIndexUpdate(size_t n = 1) {
        if (!impl_)
            return;

        // SVS index instance should not be empty
        if (indexSize() == 0) {
            this->impl_.reset();
            changes_num = 0;
            return;
        }

        changes_num += n;
        // consolidate index if number of changes bigger than 50% of index size
        const float consolidation_threshold = .5f;
        // indexSize() should not be 0 see above lines
        assert(indexSize() > 0);
        if (static_cast<float>(changes_num) / indexSize() > consolidation_threshold) {
            impl_->consolidate();
            changes_num = 0;
        }
    }

    bool isTwoLevelLVQ(const VecSimSvsQuantBits &qbits) {
        switch (qbits) {
        case VecSimSvsQuant_4x4:
        case VecSimSvsQuant_4x8:
        case VecSimSvsQuant_4x8_LeanVec:
        case VecSimSvsQuant_8x8_LeanVec:
            return true;
        default:
            return false;
        }
    }

public:
    SVSIndex(const SVSParams &params, const AbstractIndexInitParams &abstractInitParams,
             const index_component_t &components, bool force_preprocessing)
        : Base{abstractInitParams, components}, forcePreprocessing{force_preprocessing},
          changes_num{0}, buildParams{svs_details::makeVamanaBuildParameters(params)},
          search_window_size{svs_details::getOrDefault(params.search_window_size,
                                                       SVS_VAMANA_DEFAULT_SEARCH_WINDOW_SIZE)},
          search_buffer_capacity{
              svs_details::getOrDefault(params.search_buffer_capacity, search_window_size)},
          leanvec_dim{
              svs_details::getOrDefault(params.leanvec_dim, SVS_VAMANA_DEFAULT_LEANVEC_DIM)},
          epsilon{svs_details::getOrDefault(params.epsilon, SVS_VAMANA_DEFAULT_EPSILON)},
          is_two_level_lvq{isTwoLevelLVQ(params.quantBits)},
          threadpool_{std::max(size_t{SVS_VAMANA_DEFAULT_NUM_THREADS}, params.num_threads)},
          impl_{nullptr} {
        logger_ = makeLogger();
    }

    ~SVSIndex() = default;

    size_t indexSize() const override { return impl_ ? impl_->size() : 0; }

    size_t indexStorageSize() const override { return impl_ ? impl_->view_data().size() : 0; }

    size_t indexCapacity() const override {
        return impl_ ? storage_traits_t::storage_capacity(impl_->view_data()) : 0;
    }

    size_t indexLabelCount() const override {
        if constexpr (isMulti) {
            return impl_ ? impl_->labelcount() : 0;
        } else {
            return indexSize();
        }
    }

    vecsim_stl::set<size_t> getLabelsSet() const override {
        vecsim_stl::set<size_t> labels(this->allocator);
        if (impl_) {
            impl_->on_ids([&labels](size_t label) { labels.insert(label); });
        }
        return labels;
    }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_SVS;
        info.isTiered = false;
        return info;
    }

    VecSimIndexDebugInfo debugInfo() const override {
        VecSimIndexDebugInfo info;
        info.commonInfo = this->getCommonInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_SVS;

        info.svsInfo =
            svsInfoStruct{.quantBits = getCompressionMode(),
                          .alpha = this->buildParams.alpha,
                          .graphMaxDegree = this->buildParams.graph_max_degree,
                          .constructionWindowSize = this->buildParams.window_size,
                          .maxCandidatePoolSize = this->buildParams.max_candidate_pool_size,
                          .pruneTo = this->buildParams.prune_to,
                          .useSearchHistory = this->buildParams.use_full_search_history,
                          .numThreads = this->getNumThreads(),
                          .numberOfMarkedDeletedNodes = this->changes_num,
                          .searchWindowSize = this->search_window_size,
                          .searchBufferCapacity = this->search_buffer_capacity,
                          .leanvecDim = this->leanvec_dim,
                          .epsilon = this->epsilon};
        return info;
    }

    VecSimDebugInfoIterator *debugInfoIterator() const override {
        VecSimIndexDebugInfo info = this->debugInfo();
        // For readability. Update this number when needed.
        size_t numberOfInfoFields = 10;
        VecSimDebugInfoIterator *infoIterator =
            new VecSimDebugInfoIterator(numberOfInfoFields, this->allocator);

        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::ALGORITHM_STRING,
            .fieldType = INFOFIELD_STRING,
            .fieldValue = {
                FieldValue{.stringValue = VecSimAlgo_ToString(info.commonInfo.basicInfo.algo)}}});
        this->addCommonInfoToIterator(infoIterator, info.commonInfo);
        infoIterator->addInfoField(VecSim_InfoField{
            .fieldName = VecSimCommonStrings::BLOCK_SIZE_STRING,
            .fieldType = INFOFIELD_UINT64,
            .fieldValue = {FieldValue{.uintegerValue = info.commonInfo.basicInfo.blockSize}}});
        return infoIterator;
    }

    int addVector(const void *vector_data, labelType label) override {
        return addVectorsImpl(vector_data, &label, 1);
    }

    int addVectors(const void *vectors_data, const labelType *labels, size_t n) override {
        return addVectorsImpl(vectors_data, labels, n);
    }

    int deleteVector(labelType label) override { return deleteVectorsImpl(&label, 1); }

    int deleteVectors(const labelType *labels, size_t n) override {
        return deleteVectorsImpl(labels, n);
    }

    size_t getNumThreads() const override { return threadpool_.size(); }
    void setNumThreads(size_t numThreads) override { threadpool_.resize(numThreads); }

    size_t getThreadPoolCapacity() const override { return threadpool_.capacity(); }

    bool isCompressed() const override { return storage_traits_t::is_compressed(); }

    VecSimSvsQuantBits getCompressionMode() const {
        return storage_traits_t::get_compression_mode();
    }

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        if (!impl_ || !impl_->has_id(label)) {
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto query_datum = std::span{static_cast<const DataType *>(vector_data), this->dim};
        auto dist = impl_->get_distance(label, query_datum);
        return toVecSimDistance(dist);
    }

    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override {
        auto rep = new VecSimQueryReply(this->allocator);
        this->lastMode = STANDARD_KNN;
        if (k == 0 || this->indexSize() == 0) {
            return rep;
        }

        // limit result size to index size
        k = std::min(k, this->indexLabelCount());

        auto processed_query_ptr = this->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();

        auto query = svs::data::ConstSimpleDataView<DataType>{
            static_cast<const DataType *>(processed_query), 1, this->dim};
        auto result = svs::QueryResult<size_t>{query.size(), k};
        auto sp = svs_details::joinSearchParams(impl_->get_search_parameters(), queryParams,
                                                is_two_level_lvq);

        auto timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
        auto cancel = [timeoutCtx]() { return VECSIM_TIMEOUT(timeoutCtx); };

        impl_->search(result.view(), query, sp, cancel);
        if (cancel()) {
            rep->code = VecSim_QueryReply_TimedOut;
            return rep;
        }

        assert(result.n_queries() == 1);

        const auto n_neighbors = result.n_neighbors();
        rep->results.reserve(n_neighbors);

        for (size_t i = 0; i < n_neighbors; i++) {
            rep->results.push_back(
                VecSimQueryResult{result.index(0, i), toVecSimDistance(result.distance(0, i))});
        }
        // Workaround for VecSim merge_results() that expects results to be sorted
        // by score, then by id from both indices.
        // TODO: remove this workaround when merge_results() is fixed.
        sort_results_by_score_then_id(rep);
        return rep;
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const override {
        auto rep = new VecSimQueryReply(this->allocator);
        this->lastMode = RANGE_QUERY;
        if (radius == 0 || this->indexSize() == 0) {
            return rep;
        }

        auto timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
        auto cancel = [timeoutCtx]() { return VECSIM_TIMEOUT(timeoutCtx); };

        // Prepare query blob for SVS
        auto processed_query_ptr = this->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();
        std::span<const data_type> query{static_cast<const data_type *>(processed_query),
                                         this->dim};

        // Base search parameters for the SVS iterator schedule.
        auto sp = svs_details::joinSearchParams(impl_->get_search_parameters(), queryParams,
                                                is_two_level_lvq);
        // SVS BatchIterator handles the search in batches
        // The batch size is set to the index search window size by default
        const size_t batch_size = sp.buffer_config_.get_search_window_size();

        // Create SVS BatchIterator for range search
        // Search result is cached in the iterator and can be accessed by the user
        auto svs_it = impl_->make_batch_iterator(query);
        svs_it.next(batch_size, cancel);
        if (cancel()) {
            rep->code = VecSim_QueryReply_TimedOut;
            return rep;
        }

        // range search using epsilon
        const auto epsilon = queryParams && queryParams->svsRuntimeParams.epsilon != 0
                                 ? queryParams->svsRuntimeParams.epsilon
                                 : this->epsilon;

        const auto range_search_boundaries = radius * (1.0 + std::abs(epsilon));
        bool keep_searching = true;

        // Loop while iterator cache is not empty and search radius + epsilon is not exceeded
        while (keep_searching && svs_it.size() > 0) {
            // Iterate over the cached search results
            for (auto &neighbor : svs_it) {
                const auto dist = toVecSimDistance(neighbor.distance());
                if (dist <= radius) {
                    rep->results.push_back(VecSimQueryResult{neighbor.id(), dist});
                } else if (dist > range_search_boundaries) {
                    keep_searching = false;
                }
            }
            // If search radius + epsilon is not exceeded, request SVS BatchIterator for the next
            // batch
            if (keep_searching) {
                svs_it.next(batch_size, cancel);
                if (cancel()) {
                    rep->code = VecSim_QueryReply_TimedOut;
                    return rep;
                }
            }
        }
        // Workaround for VecSim merge_results() that expects results to be sorted
        // by score, then by id from both indices.
        // TODO: remove this workaround when merge_results() is fixed.
        sort_results_by_score_then_id(rep);
        return rep;
    }

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        // force_copy == true.
        auto queryBlobCopy = this->preprocessQuery(queryBlob, true);

        // take ownership of the blob copy and pass it to the batch iterator.
        auto *queryBlobCopyPtr = queryBlobCopy.release();
        // Ownership of queryBlobCopy moves to VecSimBatchIterator that will free it at the end.
        if (indexSize() == 0) {
            return new (this->getAllocator())
                NullSVS_BatchIterator(queryBlobCopyPtr, queryParams, this->getAllocator());
        } else {
            return new (this->getAllocator()) SVS_BatchIterator<impl_type, data_type>(
                queryBlobCopyPtr, impl_.get(), queryParams, this->getAllocator(), is_two_level_lvq);
        }
    }

    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        size_t index_size = this->indexSize();

        // Calculate the ratio of the subset size to the total index size.
        double subsetRatio = (index_size == 0) ? 0.f : static_cast<double>(subsetSize) / index_size;

        // Heuristic thresholds
        const double smallSubsetThreshold = 0.07;  // Subset is small if less than 7% of index.
        const double largeSubsetThreshold = 0.21;  // Subset is large If more than 21% of index.
        const double smallIndexThreshold = 75000;  // Index is small if size is less than 75k.
        const double largeIndexThreshold = 750000; // Index is large if size is more than 750k.

        bool res = false;
        if (subsetRatio < smallSubsetThreshold) {
            // For small subsets, ad-hoc if index is not large.
            res = (index_size < largeIndexThreshold);
        } else if (subsetRatio < largeSubsetThreshold) {
            // For medium subsets, ad-hoc if index is small or k is big.
            res = (index_size < smallIndexThreshold) || (k > 12);
        } else {
            // For large subsets, ad-hoc only if index is small.
            res = (index_size < smallIndexThreshold);
        }

        this->lastMode =
            res ? (initial_check ? HYBRID_ADHOC_BF : HYBRID_BATCHES_TO_ADHOC_BF) : HYBRID_BATCHES;
        return res;
    }

    void runGC() override {
        if (impl_) {
            // There is documentation for consolidate():
            // https://intel.github.io/ScalableVectorSearch/python/dynamic.html#svs.DynamicVamana.consolidate
            impl_->consolidate();
            // There is documentation for compact():
            // https://intel.github.io/ScalableVectorSearch/python/dynamic.html#svs.DynamicVamana.compact
            impl_->compact();
        }
        changes_num = 0;
    }

#ifdef BUILD_TESTS

    void saveIndexIMP(std::ofstream &output)  override;
    void impl_save(const std::string &location) override;

    // void loadIndex(const std::string &folder_path) override {
    //     svs::threads::ThreadPoolHandle threadpool_handle{VecSimSVSThreadPool{threadpool_}};
    //     // TODO rebase on master and use `logger_` field.
    //     // auto logger = makeLogger();

    //     if constexpr (isMulti) {
    //         auto loaded = svs::index::vamana::auto_multi_dynamic_assemble(
    //             folder_path + "/config",
    //             SVS_LAZY(graph_builder_t::load(folder_path + "/graph", this->blockSize,
    //                                            this->buildParams, this->getAllocator())),
    //             SVS_LAZY(storage_traits_t::load(folder_path + "/data", this->blockSize, this->dim,
    //                                             this->getAllocator())),
    //             distance_f(), std::move(threadpool_handle),
    //             svs::index::vamana::MultiMutableVamanaLoad::FROM_MULTI, logger_);
    //         impl_ = std::make_unique<impl_type>(std::move(loaded));
    //     } else {
    //         auto loaded = svs::index::vamana::auto_dynamic_assemble(
    //             folder_path + "/config",
    //             SVS_LAZY(graph_builder_t::load(folder_path + "/graph", this->blockSize,
    //                                            this->buildParams, this->getAllocator())),
    //             SVS_LAZY(storage_traits_t::load(folder_path + "/data", this->blockSize, this->dim,
    //                                             this->getAllocator())),
    //             distance_f(), std::move(threadpool_handle), false, logger_);
    //         impl_ = std::make_unique<impl_type>(std::move(loaded));
    //     }
    // }

    void fitMemory() override {}
    std::vector<std::vector<char>> getStoredVectorDataByLabel(labelType label) const override {
        assert(false && "Not implemented");
        return {};
    }
    void getDataByLabel(
        labelType label,
        std::vector<std::vector<svs_details::vecsim_dt<DataType>>> &vectors_output) const override {
        assert(false && "Not implemented");
    }

    svs::logging::logger_ptr getLogger() const override { return logger_; }
#endif
};


#ifdef BUILD_TESTS
// Including implementations for Serializer base
#include "svs_serializer_impl.h"
#endif
