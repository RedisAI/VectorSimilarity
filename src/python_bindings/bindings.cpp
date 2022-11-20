/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "VecSim/vec_sim.h"
#include "VecSim/algorithms/ngt/ngt.h"
// #include "VecSim/algorithms/hnsw/serialization.h"
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
    std::shared_ptr<VecSimBatchIterator> batchIterator;

public:
    PyBatchIterator(VecSimBatchIterator *batchIterator)
        : batchIterator(batchIterator, VecSimBatchIterator_Free) {}

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

class PyVecSimIndex {
public:
    PyVecSimIndex() {}

    PyVecSimIndex(const VecSimParams &params) { index = VecSimIndex_New(&params); }

    void addVector(py::object input, size_t id) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        VecSimIndex_AddVector(index, (void *)items.data(0), id);
    }

    void deleteVector(size_t id) { VecSimIndex_DeleteVector(index, id); }

    py::object knn(py::object input, size_t k, VecSimQueryParams *query_params) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        VecSimQueryResult_List res =
            VecSimIndex_TopKQuery(index, (void *)items.data(0), k, query_params, BY_SCORE);
        if (VecSimQueryResult_Len(res) != k) {
            throw std::runtime_error("Cannot return the results in a contiguous 2D array. Probably "
                                     "ef or M is too small");
        }
        return wrap_results(res, k);
    }

    py::object range(py::object input, double radius, VecSimQueryParams *query_params) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
        VecSimQueryResult_List res =
            VecSimIndex_RangeQuery(index, (void *)items.data(0), radius, query_params, BY_SCORE);
        return wrap_results(res, VecSimQueryResult_Len(res));
    }

    size_t indexSize() { return VecSimIndex_IndexSize(index); }

    PyBatchIterator createBatchIterator(py::object &query_blob, VecSimQueryParams *query_params) {
        py::array_t<float, py::array::c_style | py::array::forcecast> items(query_blob);
        float *vector_data = (float *)items.data(0);
        return PyBatchIterator(VecSimBatchIterator_New(index, vector_data, query_params));
    }

    virtual ~PyVecSimIndex() { VecSimIndex_Free(index); }

protected:
    VecSimIndex *index;
};

// Currently supports only floats. TODO change after serializer refactoring
class PyNGTLibIndex : public PyVecSimIndex {
public:
    PyNGTLibIndex(const NGTParams &ngt_params) {
        VecSimParams params = {.algo = VecSimAlgo_NGT, .ngtParams = ngt_params};
        this->index = VecSimIndex_New(&params);
    }

    void setDefaultEf(size_t ef) {
        auto *ngt = reinterpret_cast<NGTIndex<float, float> *>(index);
        ngt->setEf(ef);
    }
    // void saveIndex(const std::string &location) {
    //     auto serializer = HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>
    //     *>(index)); serializer.saveIndex(location);
    // }

    // void loadIndex(const std::string &location) {
    //     auto serializer = HNSWIndexSerializer(reinterpret_cast<HNSWIndex<float, float>
    //     *>(index)); serializer.loadIndex(location);
    // }
};

class PyBFIndex : public PyVecSimIndex {
public:
    PyBFIndex(const BFParams &bf_params) {
        VecSimParams params = {.algo = VecSimAlgo_BF, .bfParams = bf_params};
        this->index = VecSimIndex_New(&params);
    }
};

PYBIND11_MODULE(VecSim, m) {
    py::enum_<VecSimAlgo>(m, "VecSimAlgo")
        .value("VecSimAlgo_HNSWLIB", VecSimAlgo_HNSWLIB)
        .value("VecSimAlgo_BF", VecSimAlgo_BF)
        .value("VecSimAlgo_NGT", VecSimAlgo_NGT)
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

    py::class_<NGTParams>(m, "NGTParams")
        .def(py::init())
        .def_readwrite("type", &NGTParams::type)
        .def_readwrite("dim", &NGTParams::dim)
        .def_readwrite("metric", &NGTParams::metric)
        .def_readwrite("multi", &NGTParams::multi)
        .def_readwrite("initialCapacity", &NGTParams::initialCapacity)
        .def_readwrite("M", &NGTParams::M)
        .def_readwrite("maxPerLeaf", &NGTParams::maxPerLeaf)
        .def_readwrite("efConstruction", &NGTParams::efConstruction)
        .def_readwrite("efRuntime", &NGTParams::efRuntime)
        .def_readwrite("epsilon", &NGTParams::epsilon);

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
        .def_readwrite("ngtParams", &VecSimParams::ngtParams)
        .def_readwrite("bfParams", &VecSimParams::bfParams);

    py::class_<VecSimQueryParams> queryParams(m, "VecSimQueryParams");

    queryParams.def(py::init<>())
        .def_readwrite("ngtRuntimeParams", &VecSimQueryParams::ngtRuntimeParams);

    py::class_<NGTRuntimeParams>(queryParams, "NGTRuntimeParams")
        .def(py::init<>())
        .def_readwrite("efRuntime", &NGTRuntimeParams::efRuntime)
        .def_readwrite("epsilon", &NGTRuntimeParams::epsilon);

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

    py::class_<PyNGTLibIndex, PyVecSimIndex>(m, "NGTIndex")
        .def(py::init([](const NGTParams &params) { return new PyNGTLibIndex(params); }),
             py::arg("params"))
        .def("set_ef", &PyNGTLibIndex::setDefaultEf);
    // .def("save_index", &PyNGTLibIndex::saveIndex)
    // .def("load_index", &PyNGTLibIndex::loadIndex);

    py::class_<PyBFIndex, PyVecSimIndex>(m, "BFIndex")
        .def(py::init([](const BFParams &params) { return new PyBFIndex(params); }),
             py::arg("params"));

    py::class_<PyBatchIterator>(m, "BatchIterator")
        .def("has_next", &PyBatchIterator::hasNext)
        .def("get_next_results", &PyBatchIterator::getNextResults)
        .def("reset", &PyBatchIterator::reset);
}
