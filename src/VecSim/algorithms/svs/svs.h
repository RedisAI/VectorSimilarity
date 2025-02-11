/* TODO: change the copyright here */

/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

/* TODO clean the includes */
#pragma once
#include "VecSim/vec_sim_index.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/utils/vecsim_stl.h"
#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/spaces/spaces.h"
#include "VecSim/query_result_definitions.h"
#include "VecSim/utils/vec_utils.h"

#include <cstring>
#include <cmath>
#include <memory>
#include <queue>
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
};

template <typename MetricType, typename DataType, size_t QuantBits, size_t ResidualBits = 0>
class SVSIndex : public VecSimIndexAbstract<details::vecsim_dt<DataType>, float>,
                 public SVSIndexBase {
protected:
    using data_type = DataType;
    using distance_f = MetricType;
    using Base = VecSimIndexAbstract<details::vecsim_dt<DataType>, float>;
    using index_component_t = IndexComponents<details::vecsim_dt<DataType>, float>;

    using storage_traits_t = SVSStorageTraits<DataType, QuantBits, ResidualBits>;
    using index_storage_type = typename storage_traits_t::index_storage_type;

    using graph_builder_t = SVSGraphBuilder<uint32_t>;
    using graph_type = typename graph_builder_t::graph_type;

    using impl_type =
        svs::index::vamana::MutableVamanaIndex<graph_type, index_storage_type, distance_f>;

    size_t changes_num = 0;
    SVSParams params_;
    std::shared_ptr<impl_type> impl_; // Ownership to be shared with batch iterator

    static float toVecSimDistance(float v) { return details::toVecSimDistance<distance_f>(v); }

    static constexpr SVSParams initParams(const SVSParams &hint) {
        // clang-format off
        // TODO(rfsaliev) evaluate optimal default parameters
        // current assumption:
        // * graph_max_degree (64): =~ HNSW_M * 2; may be 63 for alignment?
        // * construction_window_size (250): =~ HNSW_EF_CONSTRUCTION
        // * max_candiate_pool_size (750): = windos_size_construction * 3
        // * prune_to (60): < graph_max_degree, optimal = graph_max_degree - 4
        // * num_threads: = CPU cores per socket
        // * search_window_size: 10 =~ HNSW_EF_RUNTIME
        #define GET_WITH_DEFAULT(v, d) ((v)?(v):(d))
        const auto construction_window_size = GET_WITH_DEFAULT(hint.construction_window_size, 250);
        const auto graph_degree = GET_WITH_DEFAULT(hint.graph_max_degree, 64);
        return SVSParams {
            .type = hint.type,
            .dim = hint.dim,
            .metric = hint.metric,
            .blockSize = hint.blockSize ? hint.blockSize : DEFAULT_BLOCK_SIZE,

            .alpha = GET_WITH_DEFAULT(hint.alpha, (hint.metric == VecSimMetric_L2 ? 1.2f : 0.9f)),
            .graph_max_degree = graph_degree,
            .construction_window_size = construction_window_size,
            .max_candidate_pool_size = GET_WITH_DEFAULT(hint.max_candidate_pool_size, construction_window_size * 3),
            .prune_to = GET_WITH_DEFAULT(hint.prune_to, graph_degree - 4),
            .use_search_history = hint.use_search_history != VecSimOption_DEFAULT ? hint.use_search_history : VecSimOption_ENABLE,
            .num_threads = GET_WITH_DEFAULT(hint.num_threads, std::thread::hardware_concurrency()),
            .search_window_size = GET_WITH_DEFAULT(hint.search_window_size, 10),
            .epsilon = hint.epsilon > 0.0 ? hint.epsilon : 0.01,
        };
        #undef GET_WITH_DEFAULT
        // clang-format on
    }

    static AbstractIndexInitParams initBaseParams(const VecSimParams *params,
                                                  std::shared_ptr<VecSimAllocator> allocator) {
        assert(params && params->algo == VecSimAlgo_SVS);
        auto &svsParams = params->algoParams.svsParams;
        size_t dataSize = VecSimParams_GetDataSize(svsParams.type, svsParams.dim, svsParams.metric);
        return {.allocator = std::move(allocator),
                .dim = svsParams.dim,
                .vecType = svsParams.type,
                .dataSize = dataSize,
                .metric = svsParams.metric,
                .blockSize = svsParams.blockSize,
                .multi = false,
                .logCtx = params->logCtx};
    }

    static svs::index::vamana::VamanaBuildParameters
    makeVamanaBuildParameters(const SVSParams &params) {
        return {params.alpha,
                params.graph_max_degree,
                params.construction_window_size,
                params.max_candidate_pool_size,
                params.prune_to,
                params.use_search_history != VecSimOption_DISABLE};
    }

    void initImpl(impl_type::data_type data, std::span<const labelType> ids) {
        svs::threads::SwitchNativeThreadPool threadpool{params_.num_threads};
        // Compute the entry point.
        auto entry_point = svs::index::vamana::extensions::compute_entry_point(data, threadpool);

        // Perform graph construction.
        auto distance = distance_f{};
        auto parameters = makeVamanaBuildParameters(params_);

        auto bs = params_.blockSize > 0 ? params_.blockSize : DEFAULT_BLOCK_SIZE;
        auto graph = graph_builder_t::build_graph(parameters, data, distance, threadpool,
                                                  entry_point, bs, this->getAllocator());

        impl_ = std::make_shared<impl_type>(std::move(graph), std::move(data), entry_point,
                                            std::move(distance), ids, std::move(threadpool));

        // Set MutableIndex build parameters
        impl_->set_construction_window_size(parameters.window_size);
        impl_->set_max_candidates(parameters.max_candidate_pool_size);
        impl_->set_prune_to(parameters.prune_to);
        impl_->set_alpha(parameters.alpha);
        impl_->set_full_search_history(parameters.use_full_search_history);

        // Configure default search parameters
        auto sp = impl_->get_search_parameters();
        sp.buffer_config({params_.search_window_size});
        impl_->set_search_parameters(sp);
        impl_->reset_performance_parameters();
    }

    int addVectorsImpl(const void *vectors_data, const labelType *labels, size_t n) {
        if (n == 0) {
            return 0;
        }

        const auto deleted_num = deleteVectorsImpl(labels, n);

        std::span<const labelType> ids(labels, n);
        auto processed_blob = this->preprocessForStorage(vectors_data);
        auto typed_vectors_data = reinterpret_cast<DataType *>(processed_blob.get());
        auto points = svs::data::SimpleDataView<DataType>{typed_vectors_data, n, params_.dim};

        // construct SVS index for first rows
        if (!impl_) {
            auto bs = params_.blockSize > 0 ? params_.blockSize : DEFAULT_BLOCK_SIZE;
            auto init_data = storage_traits_t::create_storage(points, bs, this->getAllocator());
            initImpl(std::move(init_data), ids);
            return n;
        }

        impl_->add_points(points, ids);
        return n - deleted_num;
    }

    int deleteVectorsImpl(const labelType *labels, size_t n) {
        if (indexSize() == 0) {
            return 0;
        }

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

        impl_->delete_entries(entries_to_delete);
        this->markIndexUpdate(entries_to_delete.size());
        return entries_to_delete.size();
    }

    void markIndexUpdate(size_t n = 1) {
        if (!impl_)
            return;

        if (indexSize() == 0) {
            this->impl_.reset();
            changes_num = 0;
            return;
        }

        changes_num += n;
        // consolidate index if number of changes bigger than 50% of index size
        const float consolidation_threshold = .5f;
        // indexSize() should not be be 0 see above lines
        assert(indexSize() > 0);
        if (static_cast<float>(changes_num) / indexSize() > consolidation_threshold) {
            impl_->consolidate();
            changes_num = 0;
        }
    }

public:
    SVSIndex(const VecSimParams *params, std::shared_ptr<VecSimAllocator> allocator,
             const index_component_t &components)
        : Base{initBaseParams(params, std::move(allocator)), components}, changes_num{0},
          params_{initParams(params->algoParams.svsParams)}, impl_{nullptr} {}

    ~SVSIndex() = default;

    size_t indexSize() const override { return impl_ ? impl_->size() : 0; }

    size_t indexCapacity() const override { return indexSize() + 1; }

    size_t indexLabelCount() const override { return indexSize(); }

    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_SVS;
        info.isTiered = false;
        return info;
    }

    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = this->getCommonInfo();
        info.commonInfo.basicInfo.algo = VecSimAlgo_SVS;
        info.commonInfo.basicInfo.isTiered = false;
        return info;
    }

    VecSimInfoIterator *infoIterator() const override {
        VecSimIndexInfo info = this->info();
        // For readability. Update this number when needed.
        size_t numberOfInfoFields = 10;
        VecSimInfoIterator *infoIterator =
            new VecSimInfoIterator(numberOfInfoFields, this->allocator);

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

    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        if (!impl_ || !impl_->has_id(label)) {
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto my_datum = impl_->get_datum(label);

        auto dist_f = svs::index::vamana::extensions::single_search_setup(
            impl_->view_data(), impl_->distance_function());

        auto query_datum = std::span{reinterpret_cast<const DataType *>(vector_data), params_.dim};

        svs::distance::maybe_fix_argument(dist_f, query_datum);

        auto dist = svs::distance::compute(dist_f, query_datum, my_datum);
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

        auto queries = svs::data::ConstSimpleDataView<DataType>{
            reinterpret_cast<const DataType *>(processed_query), 1, params_.dim};
        auto result = svs::QueryResult<size_t>{queries.size(), k};
        auto sp = details::joinSearchParams(impl_->get_search_parameters(), queryParams);

        auto timeoutCtx = queryParams ? queryParams->timeoutCtx : nullptr;
        auto cancel = [timeoutCtx]() { return VECSIM_TIMEOUT(timeoutCtx); };

        impl_->search(result.view(), queries, sp, cancel);
        if (cancel()) {
            rep->code = VecSim_QueryReply_TimedOut;
            return rep;
        }

        assert(result.n_queries() == 1);

        rep->results.reserve(result.n_neighbors());

        for (size_t i = 0; i < result.n_neighbors(); i++) {
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

        auto sp = details::joinSearchParams(impl_->get_search_parameters(), queryParams);
        const size_t batch_size = queryParams && queryParams->batchSize
                                      ? queryParams->batchSize
                                      : sp.buffer_config_.get_search_window_size();
        // Base search parameters for the iterator schedule.
        auto schedule = svs::index::vamana::DefaultSchedule{sp, batch_size};

        auto processed_query_ptr = this->preprocessQuery(queryBlob);
        const void *processed_query = processed_query_ptr.get();

        std::span<const data_type> query{reinterpret_cast<const data_type *>(processed_query),
                                         params_.dim};
        svs::index::vamana::BatchIterator<impl_type, data_type> svs_it{*impl_, query, schedule,
                                                                       cancel};
        if (cancel()) {
            rep->code = VecSim_QueryReply_TimedOut;
            return rep;
        }

#if 1
        // fast range search using epsilon
        const auto epsilon = queryParams && queryParams->svsRuntimeParams.epsilon > 0.0
                                 ? queryParams->svsRuntimeParams.epsilon
                                 : params_.epsilon;

        const auto range_search_boundaries = radius * (1.0 + epsilon);
        bool keep_searching = true;
        while (keep_searching && svs_it.size() > 0) {
            for (auto &neighbor : svs_it) {
                const auto dist = toVecSimDistance(neighbor.distance());
                if (dist <= radius) {
                    rep->results.emplace_back(neighbor.id(), dist);
                } else if (dist > range_search_boundaries) {
                    keep_searching = false;
                    break;
                }
            }
            if (keep_searching) {
                svs_it.next(cancel);
                if (cancel()) {
                    rep->code = VecSim_QueryReply_TimedOut;
                    return rep;
                }
            }
        }
#else
        // strict range search assuming that batch iterator results are not sorted in 100%
        int batch_times = 3;
        bool done = false;
        while (svs_it.size() > 0 && batch_times > 0) {
            for (auto &neighbor : svs_it) {
                if (toVecSimDistance(neighbor.distance()) <= radius) {
                    rep->results.emplace_back(neighbor.id(), toVecSimDistance(neighbor.distance()));
                    done = false;
                } else {
                    done = true;
                }
            }
            if (done)
                if (--batch_times == 0)
                    break;
            svs_it.next(cancel);
            if (cancel()) {
                rep->code = VecSim_QueryReply_TimedOut;
                return rep;
            }
        }
#endif
        return rep;
    }

    VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                          VecSimQueryParams *queryParams) const override {
        auto *queryBlobCopy =
            this->allocator->allocate_aligned(this->dataSize, this->preprocessors->getAlignment());
        memcpy(queryBlobCopy, queryBlob, this->dim * sizeof(DataType));
        this->preprocessQueryInPlace(queryBlobCopy);
        // Ownership of queryBlobCopy moves to VecSimBatchIterator that will free it at the end.
        if (indexSize() == 0) {
            return new (this->getAllocator())
                NullSVS_BatchIterator(queryBlobCopy, queryParams, this->getAllocator());
        } else {
            // TODO control impl_ lifetime (shared_ptr?)
            return new (this->getAllocator()) SVS_BatchIterator<impl_type, data_type>(
                queryBlobCopy, impl_, queryParams, this->getAllocator());
        }
    }

    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        bool res = true;
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
    virtual void fitMemory() {};
#endif
};
