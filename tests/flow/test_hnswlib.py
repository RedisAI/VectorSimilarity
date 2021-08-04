from VecSim import *
import numpy as np
import hnswlib
from  numpy.testing import assert_array_equal

def test_hnswlib_index():
    dim = 16
    num_elements = 10000
    space = 'l2'
    M=16
    efConstruction = 100
    efRuntime = 10


    params = VecSimParams()
    hnswparams = HNSWParams()

    params.algo = VecSimAlgo_HNSWLIB
    params.dim = dim
    params.metric = VecSimMetric_L2
    params.type = VecSimType_FLOAT32

    hnswparams.M = M
    hnswparams.efConstruction = efConstruction
    hnswparams.initialCapacity = num_elements
    hnswparams.efRuntime=efRuntime

    params.hnswParams = hnswparams
    index = VecSimIndex(params)

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    data = np.float32(np.random.random((num_elements, dim)))
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    for vector in data:
            hnswlib_labels, hnswlib_distances = p.knn_query(vector, k=10)
            redis_labels, redis_distances = index.knn_query(vector, 10)
            assert_array_equal(hnswlib_labels, redis_labels)
            assert_array_equal(hnswlib_distances, redis_distances)
