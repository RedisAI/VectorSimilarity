from VecSim import *
import numpy as np
import hnswlib
from scipy import spatial
from  numpy.testing import assert_allclose


# compare results with the original version of hnswlib - do not use elements deletion.
def test_sanity_hnswlib_index():
    dim = 16
    num_elements = 10000
    space = 'l2'
    M=16
    efConstruction = 100

    num_queries = 10
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

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    data = np.float32(np.random.random((num_elements, dim)))
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    query_data = np.float32(np.random.random((num_queries, dim)))
    for vector in query_data:
            hnswlib_labels, hnswlib_distances = p.knn_query(vector, k=10)
            redis_labels, redis_distances = index.knn_query(vector, 10)
            assert_allclose(hnswlib_labels, redis_labels,  rtol=1e-5, atol=0)
            assert_allclose(hnswlib_distances, redis_distances,  rtol=1e-5, atol=0)


# Validate correctness of delete implementation comparing the brute force search. We test the search recall which is not
# deterministic, but should be above a certain threshold. Note that recall is highly impacted by changing
# index parameters.
def test_recall_for_hnswlib_index_with_deletion():
    dim = 16
    num_elements = 10000
    M = 16
    efConstruction = 100

    num_queries = 10
    k=10
    efRuntime = 0

    hnswparams = HNSWParams()
    hnswparams.M = M
    hnswparams.efConstruction = efConstruction
    hnswparams.initialCapacity = num_elements
    hnswparams.efRuntime = efRuntime

    hnsw_index = HNSWIndex(hnswparams, VecSimType_FLOAT32, dim, VecSimMetric_L2)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    # delete half of the data
    for i in range(0, len(data), 2):
        hnsw_index.delete_vector(i)
    vectors = [vectors[i] for i in range(1, len(data), 2)]

    # We validate that we can increase ef with this designated API (if this won't work, recall should be very low)
    hnsw_index.set_ef(50)
    query_data = np.float32(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        keys = [key for _, key in dists[:k]]

        for label in hnswlib_labels[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct+=1
                    break

    # Measure recall
    recall = float(correct)/(k*num_queries)
    print("recall is: ", recall)
    assert(recall > 0.9)
