from VecSim import *
import numpy as np
from scipy import spatial
from  numpy.testing import assert_allclose
import time



def test_bf_cosine():
    dim = 128
    num_elements = 1000000
    k=10

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
    bfparams.blockSize = num_elements
    bfindex = BFIndex(bfparams, VecSimType_FLOAT32, dim, VecSimMetric_Cosine)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))


    dists = [(spatial.distance.cosine(query_data, vec), key) for key, vec in vectors]
    dists = sorted(dists)[:k]
    keys = [key for _, key in dists[:k]]
    dists = [dist for dist, _ in dists[:k]]
    start = time.time()
    bf_labels, bf_distances = bfindex.knn_query(query_data, k=10)
    end = time.time()
    print(f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)


def test_bf_l2():
    dim = 128
    num_elements = 1000000
    num_queries = 10
    k=10

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
    bfparams.blockSize = num_elements
    bfindex = BFIndex(bfparams, VecSimType_FLOAT32, dim, VecSimMetric_L2)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))


    dists = [(spatial.distance.euclidean(query_data, vec), key) for key, vec in vectors]
    dists = sorted(dists)[:k]
    keys = [key for _, key in dists[:k]]
    dists = [dist for dist, _ in dists[:k]]
    start = time.time()
    bf_labels, bf_distances = bfindex.knn_query(query_data, k=10)
    end = time.time()
    print(f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)

