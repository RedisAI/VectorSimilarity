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

namespace py = pybind11;

// Helper function that iterates query results and wrap them in python object -
// a tuple of two lists: (ids, scores)
py::object wrap_results(VecSimQueryResult_List res, size_t len) {
    size_t *data_numpy_l = new size_t[len];
    double *data_numpy_d = new double[len];
    VecSimQueryResult_Iterator *iterator = VecSimQueryResult_List_GetIterator(res);
    int res_ind = 0;
    while (VecSimQueryResult_IteratorHasNext(iterator)) {
        VecSimQueryResult *item = VecSimQueryResult_IteratorNext(iterator);
        int id = VecSimQueryResult_GetId(item);
        double score = VecSimQueryResult_GetScore(item);
        data_numpy_d[res_ind] = score;
        data_numpy_l[res_ind++] = id;
    }
    VecSimQueryResult_IteratorFree(iterator);
    VecSimQueryResult_Free(res);

    py::capsule free_when_done_l(data_numpy_l, [](void *f) { delete[] f; });
    py::capsule free_when_done_d(data_numpy_d, [](void *f) { delete[] f; });
    return py::make_tuple(
        py::array_t<size_t>(
            {(size_t)1, len},                       // shape
            {len * sizeof(size_t), sizeof(size_t)}, // C-style contiguous strides for double
            data_numpy_l,                           // the data pointer
            free_when_done_l),
        py::array_t<double>(
            {(size_t)1, len},                       // shape
            {len * sizeof(double), sizeof(double)}, // C-style contiguous strides for double
            data_numpy_d,                           // the data pointer
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
        VecSimQueryResult_List results =
            VecSimBatchIterator_Next(batchIterator.get(), n_res, order);
        // The number of results may be lower than n_res, if there are less than n_res remaining
        // vectors in the index that hadn't been returned yet.
        size_t actual_n_res = VecSimQueryResult_Len(results);
        return wrap_results(results, actual_n_res);
    }
    void reset() { VecSimBatchIterator_Reset(batchIterator.get()); }
    virtual ~PyBatchIterator() {}
};
// @input or @query arguments are a py::object object. (numpy arrays are acceptable)

// To convert input or query to a pointer use input_to_blob(input)
// For example:
// VecSimIndex_AddVector(index, input_to_blob(input), id);

class PyVecSimIndex {
public:
    PyVecSimIndex()
        : create_bytearray(
              py::module::import("src.python_bindings.Mybytearray").attr("create_bytearray")) {}

    PyVecSimIndex(const VecSimParams &params)
        : create_bytearray(
              py::module::import("src.python_bindings.Mybytearray").attr("create_bytearray")) {
        index = std::shared_ptr<VecSimIndex>(VecSimIndex_New(&params), VecSimIndex_Free);
    }

    void addVector(py::object input, size_t id) {
        VecSimIndex_AddVector(index.get(), input_to_blob(input), id);
    }
    void deleteVector(size_t id) { VecSimIndex_DeleteVector(index.get(), id); }

    py::object knn(py::object input, size_t k, VecSimQueryParams *query_params) {
        VecSimQueryResult_List res =
            VecSimIndex_TopKQuery(index.get(), input_to_blob(input), k, query_params, BY_SCORE);
        if (VecSimQueryResult_Len(res) != k) {
            throw std::runtime_error("Cannot return the results in a contiguous 2D array. Probably "
                                     "ef or M is too small");
        }
        return wrap_results(res, k);
    }

    py::object range(py::object input, double radius, VecSimQueryParams *query_params) {
        VecSimQueryResult_List res = VecSimIndex_RangeQuery(index.get(), input_to_blob(input),
                                                            radius, query_params, BY_SCORE);
        return wrap_results(res, VecSimQueryResult_Len(res));
    }

    size_t indexSize() { return VecSimIndex_IndexSize(index.get()); }

    PyBatchIterator createBatchIterator(py::object input, VecSimQueryParams *query_params) {
        return PyBatchIterator(
            index, VecSimBatchIterator_New(index.get(), input_to_blob(input), query_params));
    }

    virtual ~PyVecSimIndex() {} // Delete function was given to the shared pointer object

protected:
    std::shared_ptr<VecSimIndex> index;

private:
    // save the bytearray to keep its pointer valid
    py::bytearray tmp_bytearray;
    const py::function create_bytearray;
    const char *input_to_blob(py::object input) {
        tmp_bytearray = create_bytearray(input);
        return PyByteArray_AS_STRING(tmp_bytearray.ptr());
    }
};

class PyHNSWLibIndex : public PyVecSimIndex {
public:
    PyHNSWLibIndex(const HNSWParams &hnsw_params) {
        VecSimParams params = {.algo = VecSimAlgo_HNSWLIB, .hnswParams = hnsw_params};
        this->index = std::shared_ptr<VecSimIndex>(VecSimIndex_New(&params), VecSimIndex_Free);
    }

    // @params is required only in V1.
    PyHNSWLibIndex(const std::string &location, const HNSWParams *hnsw_params = nullptr) {
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
};

class PyBFIndex : public PyVecSimIndex {
public:
    PyBFIndex(const BFParams &bf_params) {
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
             py::arg("query_param") = nullptr);

    py::class_<PyHNSWLibIndex, PyVecSimIndex>(m, "HNSWIndex")
        .def(py::init([](const HNSWParams &params) { return new PyHNSWLibIndex(params); }),
             py::arg("params"))
        .def(py::init([](const std::string &location, const HNSWParams *params) {
                 return new PyHNSWLibIndex(location, params);
             }),
             py::arg("location"), py::arg("params") = nullptr)
        .def("set_ef", &PyHNSWLibIndex::setDefaultEf)
        .def("save_index", &PyHNSWLibIndex::saveIndex);

    py::class_<PyBFIndex, PyVecSimIndex>(m, "BFIndex")
        .def(py::init([](const BFParams &params) { return new PyBFIndex(params); }),
             py::arg("params"));

    py::class_<PyBatchIterator>(m, "BatchIterator")
        .def("has_next", &PyBatchIterator::hasNext)
        .def("get_next_results", &PyBatchIterator::getNextResults)
        .def("reset", &PyBatchIterator::reset);
}
