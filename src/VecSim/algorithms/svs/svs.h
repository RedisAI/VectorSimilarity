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

#include "VecSim/algorithms/svs/svs_utils.h"
#include "VecSim/algorithms/svs/svs_batch_iterator.h"
#include "VecSim/algorithms/svs/svs_extensions.h"

struct SVSIndexBase {
    virtual ~SVSIndexBase() = default;
    virtual int addVectors(const void *vectors_data, const labelType *labels, size_t n) = 0;
    virtual int deleteVectors(const labelType *labels, size_t n) = 0;
    virtual size_t indexStorageSize() const = 0;
    virtual size_t getNumThreads() const = 0;
    virtual void setNumThreads(size_t numThreads) = 0;
    virtual size_t getThreadPoolCapacity() const = 0;
};

template <typename MetricType, typename DataType, size_t QuantBits, size_t ResidualBits,
          bool IsLeanVec>
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

    using impl_type =
        svs::index::vamana::MutableVamanaIndex<graph_type, index_storage_type, distance_f>;

    bool forcePreprocessing;

    // Index severe changes counter to initiate reindexing if number of changes exceed threshold
    // markIndexUpdated() manages this counter
    size_t changes_num;

    // Index build parameters
    svs::index::vamana::VamanaBuildParameters buildParams;

    // Index search parameters
    size_t search_window_size;
    double epsilon;

    // SVS thread pool
    VecSimSVSThreadPool threadpool_;
    // SVS Index implementation instance
    std::unique_ptr<impl_type> impl_;

    static float toVecSimDistance(float v) { return svs_details::toVecSimDistance<distance_f>(v); }

    template <typename T, typename U>
    static T getOrDefault(T v, U def) {
        return v ? v : static_cast<T>(def);
    }

    static svs::index::vamana::VamanaBuildParameters
    makeVamanaBuildParameters(const SVSParams &params) {
        // clang-format off
        // evaluate optimal default parameters; current assumption:
        // * alpha (1.2 or 0.9) depends on metric: L2: > 1.0, IP, Cosine: < 1.0
        //      In the Vamana algorithm implementation in SVS, the choice of alpha value
        //      depends on the type of similarity measure used. For L2, which minimizes distance,
        //      an alpha value greater than 1 is needed, typically around 1.2.
        //      For Inner Product and Cosine, which maximize similarity or distance,
        //      the alpha value should be less than 1, usually 0.9 or 0.95 works.
        // * construction_window_size (250): similar to HNSW_EF_CONSTRUCTION
        // * graph_max_degree (64): similar to HNSW_M * 2
        // * max_candidate_pool_size (750): =~ construction_window_size * 3
        // * prune_to (60): < graph_max_degree, optimal = graph_max_degree - 4
        //      The prune_to parameter is a performance feature designed to enhance build time
        //      by setting a small difference between this value and the maximum graph degree.
        //      This acts as a threshold for how much pruning can reduce the number of neighbors.
        //      Typically, a small gap of 4 or 8 is sufficient to improve build time
        //      without compromising the quality of the graph.
        // * use_search_history (true): now: is enabled if not disabled explicitly
        //                              future: default value based on other index parameters
        const auto construction_window_size = getOrDefault(params.construction_window_size, 250);
        const auto graph_max_degree = getOrDefault(params.graph_max_degree, 64);

        // More info about VamanaBuildParameters can be found there:
        // https://intel.github.io/ScalableVectorSearch/python/vamana.html#svs.VamanaBuildParameters
        return svs::index::vamana::VamanaBuildParameters{
            getOrDefault(params.alpha, (params.metric == VecSimMetric_L2 ? 1.2f : 0.9f)),
            graph_max_degree,
            construction_window_size,
            getOrDefault(params.max_candidate_pool_size, construction_window_size * 3),
            getOrDefault(params.prune_to, graph_max_degree - 4),
            params.use_search_history != VecSimOption_DISABLE
        };
        // clang-format on
    }

    // Create SVS index instance with initial data
    // Data should not be empty
    template <svs::data::ImmutableMemoryDataset Dataset>
    void initImpl(const Dataset &points, std::span<const labelType> ids) {
        svs::threads::ThreadPoolHandle threadpool_handle{VecSimSVSThreadPool{threadpool_}};
        // Construct SVS index initial storage with compression if needed
        auto data = storage_traits_t::create_storage(points, this->blockSize, threadpool_handle,
                                                     this->getAllocator());
        // Compute the entry point.
        auto entry_point =
            svs::index::vamana::extensions::compute_entry_point(data, threadpool_handle);

        // Perform graph construction.
        auto distance = distance_f{};
        const auto &parameters = this->buildParams;

        // Construct initial Vamana Graph
        auto graph =
            graph_builder_t::build_graph(parameters, data, distance, threadpool_, entry_point,
                                         this->blockSize, this->getAllocator());

        // Create SVS MutableIndex instance
        impl_ = std::make_unique<impl_type>(std::move(graph), std::move(data), entry_point,
                                            std::move(distance), ids, threadpool_);

        // Set SVS MutableIndex build parameters to be used in future updates
        impl_->set_construction_window_size(parameters.window_size);
        impl_->set_max_candidates(parameters.max_candidate_pool_size);
        impl_->set_prune_to(parameters.prune_to);
        impl_->set_alpha(parameters.alpha);
        impl_->set_full_search_history(parameters.use_full_search_history);

        // Configure default search parameters
        auto sp = impl_->get_search_parameters();
        sp.buffer_config({this->search_window_size});
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
            this->preprocessQueryInPlace(static_cast<DataType *>(processed_blob.get()) +
                                         i * this->dim);
        }
        return processed_blob;
    }

    int addVectorsImpl(const void *vectors_data, const labelType *labels, size_t n) {
        if (n == 0) {
            return 0;
        }

        // SVS index does not support adding vectors with the same label
        // so we have to delete them first
        const auto deleted_num = deleteVectorsImpl(labels, n);

        std::span<const labelType> ids(labels, n);
        auto processed_blob = this->preprocessForBatchStorage(vectors_data, n);
        auto typed_vectors_data = static_cast<DataType *>(processed_blob.get());
        // Wrap data into SVS SimpleDataView for SVS API
        auto points = svs::data::SimpleDataView<DataType>{typed_vectors_data, n, this->dim};

        // If n == 1, we should ensure single-threading
        const size_t num_threads = (n == 1) ? getNumThreads() : 1;
        if (num_threads > 1) {
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
        if (num_threads > 1) {
            setNumThreads(num_threads);
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
        const size_t num_threads = (entries_to_delete.size() == 1) ? getNumThreads() : 1;
        if (num_threads > 1) {
            setNumThreads(1);
        }

        impl_->delete_entries(entries_to_delete);

        // Restore multi-threading if needed
        if (num_threads > 1) {
            setNumThreads(num_threads);
        }

        this->markIndexUpdate(entries_to_delete.size());
        return entries_to_delete.size();
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

public:
    SVSIndex(const SVSParams &params, const AbstractIndexInitParams &abstractInitParams,
             const index_component_t &components, bool force_preprocessing)
        : Base{abstractInitParams, components}, forcePreprocessing{force_preprocessing},
          changes_num{0}, buildParams{makeVamanaBuildParameters(params)},
          search_window_size{getOrDefault(params.search_window_size, 10)},
          epsilon{getOrDefault(params.epsilon, 0.01)},
          threadpool_{std::max(size_t{1}, params.num_threads)}, impl_{nullptr} {}

    ~SVSIndex() = default;

    size_t indexSize() const override { return impl_ ? impl_->size() : 0; }

    size_t indexStorageSize() const override { return impl_ ? impl_->view_data().size() : 0; }

    size_t indexCapacity() const override {
        return impl_ ? storage_traits_t::storage_capacity(impl_->view_data()) : 0;
    }

    size_t indexLabelCount() const override { return indexSize(); }

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
        info.commonInfo.basicInfo.isTiered = false;
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

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        if (!impl_ || !impl_->has_id(label)) {
            return std::numeric_limits<double>::quiet_NaN();
        };

        // Get SVS distance function
        auto dist_f = impl_->distance_function();
        size_t id = impl_->translate_external_id(label);
        auto query = std::span{static_cast<const DataType *>(vector_data), this->dim};

        // Depending on LVQ/LeanVec, SVS distance function may need special treatment
        float dist =
            storage_traits_t::compute_distance_by_id(impl_->view_data(), dist_f, id, query);
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
        k = std::min(k, this->indexSize());

        auto processed_query_ptr = this->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();

        auto query = svs::data::ConstSimpleDataView<DataType>{
            static_cast<const DataType *>(processed_query), 1, this->dim};
        auto result = svs::QueryResult<size_t>{query.size(), k};
        auto sp = svs_details::joinSearchParams(impl_->get_search_parameters(), queryParams);

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
            rep->results.emplace_back(result.index(0, i), toVecSimDistance(result.distance(0, i)));
        }
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
        auto sp = svs_details::joinSearchParams(impl_->get_search_parameters(), queryParams);
        // SVS BatchIterator handles the search in batches
        // The batch size is set to the index search window size by default
        const size_t batch_size = sp.buffer_config_.get_search_window_size();
        auto schedule = svs::index::vamana::DefaultSchedule{sp, batch_size};

        // Create SVS BatchIterator for range search
        // SVS BatchIterator executes first batch of search at construction
        // Search result is cached in the iterator and can be accessed by the user
        svs::index::vamana::BatchIterator<impl_type, data_type> svs_it{*impl_, query, schedule,
                                                                       cancel};
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
                    rep->results.emplace_back(neighbor.id(), dist);
                } else if (dist > range_search_boundaries) {
                    keep_searching = false;
                }
            }
            // If search radius + epsilon is not exceeded, request SVS BatchIterator for the next
            // batch
            if (keep_searching) {
                svs_it.next(cancel);
                if (cancel()) {
                    rep->code = VecSim_QueryReply_TimedOut;
                    return rep;
                }
            }
        }
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
                queryBlobCopyPtr, impl_.get(), queryParams, this->getAllocator());
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
            impl_->consolidate();
            impl_->compact();
        }
        changes_num = 0;
    }

#ifdef BUILD_TESTS
    void fitMemory() override {}
    std::vector<std::vector<char>> getStoredVectorDataByLabel(labelType label) const override {
        assert(nullptr && "Not implemented");
        return {};
    }
    void getDataByLabel(
        labelType label,
        std::vector<std::vector<svs_details::vecsim_dt<DataType>>> &vectors_output) const override {
        assert(nullptr && "Not implemented");
    }
#endif
};
