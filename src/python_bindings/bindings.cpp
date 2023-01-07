/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/hnsw/hnsw.h"
#include "VecSim/algorithms/hnsw/hnsw_factory.h"
#include "VecSim/batch_iterator.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include <cstring>
#include <thread>
#include <VecSim/algorithms/hnsw/hnsw_single.h>
#include <VecSim/algorithms/brute_force/brute_force_single.h>

namespace py = pybind11;

// Helper function that iterates query results and wrap them in python numpy object -
// a tuple of two 2D arrays: (labels, distances)
py::object wrap_results(VecSimQueryResult_List *res, size_t num_res, size_t num_queries = 1) {
    auto *data_numpy_l = new long[num_res * num_queries];
    auto *data_numpy_d = new double[num_res * num_queries];
    // Default "padding" for the entries that will stay empty (in case of less than k results return
    // in KNN, or results of range queries with number of results lower than the maximum in the
    // batch (which determines the arrays' shape)
    std::fill_n(data_numpy_l, num_res * num_queries, -1);
    std::fill_n(data_numpy_d, num_res * num_queries, -1.0);

    for (size_t i = 0; i < num_queries; i++) {
        VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res[i]);
        size_t res_ind = i * num_res;
        while (VecSimQueryResult_IteratorHasNext(iterator)) {
            VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
            size_t id = VecSimQueryResult_GetId(item);
            double score = VecSimQueryResult_GetScore(item);
            data_numpy_d[res_ind] = score;
            data_numpy_l[res_ind++] = (long)id;
        }
        VecSimQueryResult_IteratorFree(iterator);
        VecSimQueryResult_Free(res[i]);
    }

    py::capsule free_when_done_l(data_numpy_l, [](void *labels) { delete[](long *) labels; });
    py::capsule free_when_done_d(data_numpy_d, [](void *dists) { delete[](double *) dists; });
    return py::make_tuple(
        py::array_t<long>(
            {(size_t)num_queries, num_res},         // shape
            {num_res * sizeof(long), sizeof(long)}, // C-style contiguous strides for size_t
            data_numpy_l,                           // the data pointer (labels array)
            free_when_done_l),
        py::array_t<double>(
            {(size_t)num_queries, num_res},             // shape
            {num_res * sizeof(double), sizeof(double)}, // C-style contiguous strides for double
            data_numpy_d,                               // the data pointer (distances array)
            free_when_done_d));
}

class PyBatchIterator {
private:
    // Hold the index pointer, so that it will be destroyed **after** the batch iterator. Hence,
    // the index field should come before the iterator field.
    std::shared_ptr<VecSimIndex> vectorIndex;
    std::shared_ptr<VecSimBatchIterator> batchIterator;

public:
    PyBatchIterator(const std::shared_ptr<VecSimIndex> &vecIndex,
                    VecSimBatchIterator *batchIterator)
        : vectorIndex(vecIndex), batchIterator(batchIterator, VecSimBatchIterator_Free) {}

    bool hasNext() { return VecSimBatchIterator_HasNext(batchIterator.get()); }

    py::object getNextResults(size_t n_res, VecSimQueryResult_Order order) {
        VecSimQueryResult_List results;
        {
            py::gil_scoped_release py_gil;
            results = VecSimBatchIterator_Next(batchIterator.get(), n_res, order);
        }
        // The number of results may be lower than n_res, if there are less than n_res remaining
        // vectors in the index that hadn't been returned yet.
        size_t actual_n_res = VecSimQueryResult_Len(results);
        return wrap_results(&results, actual_n_res);
    }
    void reset() { VecSimBatchIterator_Reset(batchIterator.get()); }
    virtual ~PyBatchIterator() {}
};

// @input or @query arguments are a py::object object. (numpy arrays are acceptable)
class PyVecSimIndex {
private:
    template <typename DataType>
    inline py::object rawVectorsAsNumpy(const std::vector<const char *> &vectors, size_t dim) {

        size_t n_vectors = vectors.size();
        auto *data_numpy = new DataType[n_vectors * dim];
        // Copy the vector blobs into one contiguous array of data, and free the original buffer
        // afterwards.
        for (size_t i = 0; i < n_vectors; i++) {
            memcpy(data_numpy + i * dim, vectors[i], dim * sizeof(DataType));
            delete[] vectors[i];
        }

        py::capsule free_when_done(data_numpy,
                                   [](void *vector_data) { delete[](DataType *) vector_data; });
        return py::array_t<DataType>(
            {n_vectors, dim}, // shape
            {dim * sizeof(DataType),
             sizeof(DataType)}, // C-style contiguous strides for the data type
            data_numpy,         // the data pointer
            free_when_done);
    }

protected:
    std::shared_ptr<VecSimIndex> index;

    inline VecSimQueryResult_List searchKnnInternal(const char *query, size_t k,
                                                    VecSimQueryParams *query_params) {
        return VecSimIndex_TopKQuery(index.get(), query, k, query_params, BY_SCORE);
    }

    inline void addVectorInternal(const char *vector_data, size_t id) {
        VecSimIndex_AddVector(index.get(), vector_data, id);
    }

    inline VecSimQueryResult_List searchRangeInternal(const char *query, double radius,
                                                      VecSimQueryParams *query_params) {
        return VecSimIndex_RangeQuery(index.get(), query, radius, query_params, BY_SCORE);
    }

public:
    PyVecSimIndex() = default;

    explicit PyVecSimIndex(const VecSimParams &params) {
        index = std::shared_ptr<VecSimIndex>(VecSimIndex_New(&params), VecSimIndex_Free);
    }

    void addVector(const py::object &input, size_t id) {
        py::array vector_data(input);
        py::gil_scoped_release py_gil;
        addVectorInternal((const char *)vector_data.data(0), id);
    }

    void deleteVector(size_t id) { VecSimIndex_DeleteVector(index.get(), id); }

    py::object knn(const py::object &input, size_t k, VecSimQueryParams *query_params) {
        py::array query(input);
        VecSimQueryResult_List res;
        {
            py::gil_scoped_release py_gil;
            res = searchKnnInternal((const char *)query.data(0), k, query_params);
        }
        return wrap_results(&res, k);
    }

    py::object range(const py::object &input, double radius, VecSimQueryParams *query_params) {
        py::array query(input);
        VecSimQueryResult_List res;
        {
            py::gil_scoped_release py_gil;
            res = searchRangeInternal((const char *)query.data(0), radius, query_params);
        }
        return wrap_results(&res, VecSimQueryResult_Len(res));
    }

    size_t indexSize() { return VecSimIndex_IndexSize(index.get()); }

    PyBatchIterator createBatchIterator(const py::object &input, VecSimQueryParams *query_params) {
        py::array query(input);
        return PyBatchIterator(
            index, VecSimBatchIterator_New(index.get(), (const char *)query.data(0), query_params));
    }

    py::object getVector(labelType label) {
        std::vector<const char *> res;
        reinterpret_cast<VecSimIndexInterface *>(this->index.get())->getDataByLabel(label, res);
        if (index->info().algo == VecSimAlgo_BF) {
            size_t dim = index->info().bfInfo.dim;
            if (index->info().bfInfo.type == VecSimType_FLOAT32) {
                return rawVectorsAsNumpy<float>(res, dim);
            } else if (index->info().bfInfo.type == VecSimType_FLOAT64) {
                return rawVectorsAsNumpy<double>(res, dim);
            } else {
                throw std::runtime_error("Invalid vector data type");
            }
        } else if (index->info().algo == VecSimAlgo_HNSWLIB) {
            size_t dim = index->info().hnswInfo.dim;
            if (index->info().hnswInfo.type == VecSimType_FLOAT32) {
                return rawVectorsAsNumpy<float>(res, dim);
            } else if (index->info().hnswInfo.type == VecSimType_FLOAT64) {
                return rawVectorsAsNumpy<double>(res, dim);
            } else {
                throw std::runtime_error("Invalid vector data type");
            }
        } else {
            throw std::runtime_error("Invalid index algorithm");
        }
    }

    virtual ~PyVecSimIndex() = default; // Delete function was given to the shared pointer object
};

class PyHNSWLibIndex : public PyVecSimIndex {
private:
    template <typename searchParam> // size_t/double for KNN/range queries.
    using QueryFunc =
        std::function<VecSimQueryResult_List(const char *, searchParam, VecSimQueryParams *)>;

    template <typename searchParam> // size_t/double for KNN / range queries.
    void runParallelQueries(const py::array &queries, size_t n_queries, searchParam param,
                            VecSimQueryParams *query_params, int n_threads,
                            QueryFunc<searchParam> queryFunc, VecSimQueryResult_List *results) {

        // Use number of hardware cores as default number of threads, unless specified otherwise.
        if (n_threads <= 0) {
            n_threads = (int)std::thread::hardware_concurrency();
        }
        std::atomic_int global_counter(0);

        auto parallel_search = [&](const py::array &items) {
            while (true) {
                int ind = global_counter.fetch_add(1);
                if (ind >= n_queries) {
                    break;
                }
                results[ind] = queryFunc((const char *)items.data(ind), param, query_params);
            }
        };
        std::thread thread_objs[n_threads];
        {
            // Release python GIL while threads are running.
            py::gil_scoped_release py_gil;
            for (size_t i = 0; i < n_threads; i++) {
                thread_objs[i] = std::thread(parallel_search, queries);
            }
            for (size_t i = 0; i < n_threads; i++) {
                thread_objs[i].join();
            }
        }
    }

public:
    explicit PyHNSWLibIndex(const HNSWParams &hnsw_params) {
        VecSimParams params = {.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        this->index = std::shared_ptr<VecSimIndex>(VecSimIndex_New(&params), VecSimIndex_Free);
    }

    // @params is required only in V1.
    explicit PyHNSWLibIndex(const std::string &location, const HNSWParams *hnsw_params = nullptr) {
        this->index = std::shared_ptr<VecSimIndex>(HNSWFactory::NewIndex(location, hnsw_params),
                                                   VecSimIndex_Free);
    }

    void setDefaultEf(size_t ef) {
        auto *hnsw = reinterpret_cast<HNSWIndex<float, float> *>(index.get());
        hnsw->setEf(ef);
    }
    void saveIndex(const std::string &location) {
        auto *hnsw = reinterpret_cast<HNSWIndex<float, float> *>(index.get());
        hnsw->saveIndex(location);
    }
    py::object searchKnnParallel(const py::object &input, size_t k, VecSimQueryParams *query_params,
                                 int n_threads) {

        py::array queries(input);
        if (queries.ndim() != 2) {
            throw std::runtime_error("Input queries array must be 2D array");
        }
        size_t n_queries = queries.shape(0);
        std::function<VecSimQueryResult_List(const char *, size_t, VecSimQueryParams *)>
            searchKnnWrapper([this](const char *query_, size_t k_,
                                    VecSimQueryParams *query_params_) -> VecSimQueryResult_List {
                return this->searchKnnInternal(query_, k_, query_params_);
            });
        VecSimQueryResult_List results[n_queries];
        runParallelQueries<size_t>(queries, n_queries, k, query_params, n_threads, searchKnnWrapper,
                                   results);
        return wrap_results(results, k, n_queries);
    }
    py::object searchRangeParallel(const py::object &input, double radius,
                                   VecSimQueryParams *query_params, int n_threads) {
        py::array queries(input);
        if (queries.ndim() != 2) {
            throw std::runtime_error("Input queries array must be 2D array");
        }
        size_t n_queries = queries.shape(0);
        std::function<VecSimQueryResult_List(const char *, double, VecSimQueryParams *)>
            searchRangeWrapper([this](const char *query_, double radius_,
                                      VecSimQueryParams *query_params_) -> VecSimQueryResult_List {
                return this->searchRangeInternal(query_, radius_, query_params_);
            });
        VecSimQueryResult_List results[n_queries];
        runParallelQueries<double>(queries, n_queries, radius, query_params, n_threads,
                                   searchRangeWrapper, results);
        size_t max_results_num = 1;
        for (size_t i = 0; i < n_queries; i++) {
            if (VecSimQueryResult_Len(results[i]) > max_results_num) {
                max_results_num = VecSimQueryResult_Len(results[i]);
            }
        }
        // We return 2D numpy array of results (labels and distances), use padding of "-1" in the
        // empty entries of the matrices.
        return wrap_results(results, max_results_num, n_queries);
    }

    void addVectorsParallel(const py::object &input, const py::object &vectors_labels,
                            int n_threads) {
        py::array vectors_data(input);
        py::array_t<labelType, py::array::c_style | py::array::forcecast> labels(vectors_labels);

        if (vectors_data.ndim() != 2) {
            throw std::runtime_error("Input vectors data array must be 2D array");
        }
        if (labels.ndim() != 1) {
            throw std::runtime_error("Input vectors labels array must be 1D array");
        }
        if (vectors_data.shape(0) != labels.shape(0)) {
            throw std::runtime_error(
                "The first dim of vectors data and labels arrays must be equal");
        }
        size_t n_vectors = vectors_data.shape(0);
        // Use number of hardware cores as default number of threads, unless specified otherwise.
        if (n_threads <= 0) {
            n_threads = (int)std::thread::hardware_concurrency();
        }

        std::atomic_int global_counter(0);
        auto parallel_insert =
            [&](const py::array &data,
                const py::array_t<labelType, py::array::c_style | py::array::forcecast> &labels) {
                while (true) {
                    int ind = global_counter.fetch_add(1);
                    if (ind >= n_vectors) {
                        break;
                    }
                    this->addVectorInternal((const char *)data.data(ind), labels.at(ind));
                }
            };
        std::thread thread_objs[n_threads];
        {
            // Release python GIL while threads are running.
            py::gil_scoped_release py_gil;
            for (size_t i = 0; i < n_threads; i++) {
                thread_objs[i] = std::thread(parallel_insert, vectors_data, labels);
            }
            for (size_t i = 0; i < n_threads; i++) {
                thread_objs[i].join();
            }
        }
    }

    bool checkIntegrity() {
        auto type = VecSimIndex_Info(this->index.get()).hnswInfo.type;
        if (type == VecSimType_FLOAT32) {
            return reinterpret_cast<HNSWIndex<float, float> *>(this->index.get())
                ->checkIntegrity()
                .valid_state;
        } else if (type == VecSimType_FLOAT64) {
            return reinterpret_cast<HNSWIndex<double, double> *>(this->index.get())
                ->checkIntegrity()
                .valid_state;
        } else {
            throw std::runtime_error("Invalid index data type");
        }
    }
};

class PyBFIndex : public PyVecSimIndex {
public:
    explicit PyBFIndex(const BFParams &bf_params) {
        VecSimParams params = {.algo = VecSimAlgo_BF, .bfParams = bf_params};
        this->index = std::shared_ptr<VecSimIndex>(VecSimIndex_New(&params), VecSimIndex_Free);
    }
};

PYBIND11_MODULE(VecSim, m) {
    py::enum_<VecSimAlgo>(m, "VecSimAlgo")
        .value("VecSimAlgo_HNSWLIB", VecSimAlgo_HNSWLIB)
        .value("VecSimAlgo_BF", VecSimAlgo_BF)
        .export_values();

    py::enum_<VecSimType>(m, "VecSimType")
        .value("VecSimType_FLOAT32", VecSimType_FLOAT32)
        .value("VecSimType_FLOAT64", VecSimType_FLOAT64)
        .value("VecSimType_INT32", VecSimType_INT32)
        .value("VecSimType_INT64", VecSimType_INT64)
        .export_values();

    py::enum_<VecSimMetric>(m, "VecSimMetric")
        .value("VecSimMetric_L2", VecSimMetric_L2)
        .value("VecSimMetric_IP", VecSimMetric_IP)
        .value("VecSimMetric_Cosine", VecSimMetric_Cosine)
        .export_values();

    py::enum_<VecSimQueryResult_Order>(m, "VecSimQueryResult_Order")
        .value("BY_SCORE", BY_SCORE)
        .value("BY_ID", BY_ID)
        .export_values();

    py::class_<HNSWParams>(m, "HNSWParams")
        .def(py::init())
        .def_readwrite("type", &HNSWParams::type)
        .def_readwrite("dim", &HNSWParams::dim)
        .def_readwrite("metric", &HNSWParams::metric)
        .def_readwrite("multi", &HNSWParams::multi)
        .def_readwrite("initialCapacity", &HNSWParams::initialCapacity)
        .def_readwrite("M", &HNSWParams::M)
        .def_readwrite("efConstruction", &HNSWParams::efConstruction)
        .def_readwrite("efRuntime", &HNSWParams::efRuntime)
        .def_readwrite("epsilon", &HNSWParams::epsilon);

    py::class_<BFParams>(m, "BFParams")
        .def(py::init())
        .def_readwrite("type", &BFParams::type)
        .def_readwrite("dim", &BFParams::dim)
        .def_readwrite("metric", &BFParams::metric)
        .def_readwrite("multi", &BFParams::multi)
        .def_readwrite("initialCapacity", &BFParams::initialCapacity)
        .def_readwrite("blockSize", &BFParams::blockSize);

    py::class_<VecSimParams>(m, "VecSimParams")
        .def(py::init())
        .def_readwrite("algo", &VecSimParams::algo)
        .def_readwrite("hnswParams", &VecSimParams::hnswParams)
        .def_readwrite("bfParams", &VecSimParams::bfParams);

    py::class_<VecSimQueryParams> queryParams(m, "VecSimQueryParams");

    queryParams.def(py::init<>())
        .def_readwrite("hnswRuntimeParams", &VecSimQueryParams::hnswRuntimeParams);

    py::class_<HNSWRuntimeParams>(queryParams, "HNSWRuntimeParams")
        .def(py::init<>())
        .def_readwrite("efRuntime", &HNSWRuntimeParams::efRuntime)
        .def_readwrite("epsilon", &HNSWRuntimeParams::epsilon);

    py::class_<PyVecSimIndex>(m, "VecSimIndex")
        .def(py::init([](const VecSimParams &params) { return new PyVecSimIndex(params); }),
             py::arg("params"))
        .def("add_vector", &PyVecSimIndex::addVector)
        .def("delete_vector", &PyVecSimIndex::deleteVector)
        .def("knn_query", &PyVecSimIndex::knn, py::arg("vector"), py::arg("k"),
             py::arg("query_param") = nullptr)
        .def("range_query", &PyVecSimIndex::range, py::arg("vector"), py::arg("radius"),
             py::arg("query_param") = nullptr)
        .def("index_size", &PyVecSimIndex::indexSize)
        .def("create_batch_iterator", &PyVecSimIndex::createBatchIterator, py::arg("query_blob"),
             py::arg("query_param") = nullptr)
        .def("get_vector", &PyVecSimIndex::getVector);

    py::class_<PyHNSWLibIndex, PyVecSimIndex>(m, "HNSWIndex")
        .def(py::init([](const HNSWParams &params) { return new PyHNSWLibIndex(params); }),
             py::arg("params"))
        .def(py::init([](const std::string &location, const HNSWParams *params) {
                 return new PyHNSWLibIndex(location, params);
             }),
             py::arg("location"), py::arg("params") = nullptr)
        .def("set_ef", &PyHNSWLibIndex::setDefaultEf)
        .def("save_index", &PyHNSWLibIndex::saveIndex)
        .def("knn_parallel", &PyHNSWLibIndex::searchKnnParallel, py::arg("queries"), py::arg("k"),
             py::arg("query_param") = nullptr, py::arg("num_threads") = -1)
        .def("add_vector_parallel", &PyHNSWLibIndex::addVectorsParallel, py::arg("vectors"),
             py::arg("labels"), py::arg("num_threads") = -1)
        .def("check_integrity", &PyHNSWLibIndex::checkIntegrity)
        .def("range_parallel", &PyHNSWLibIndex::searchRangeParallel, py::arg("queries"),
             py::arg("radius"), py::arg("query_param") = nullptr, py::arg("num_threads") = -1);

    py::class_<PyBFIndex, PyVecSimIndex>(m, "BFIndex")
        .def(py::init([](const BFParams &params) { return new PyBFIndex(params); }),
             py::arg("params"));

    py::class_<PyBatchIterator>(m, "BatchIterator")
        .def("has_next", &PyBatchIterator::hasNext)
        .def("get_next_results", &PyBatchIterator::getNextResults)
        .def("reset", &PyBatchIterator::reset);
}
