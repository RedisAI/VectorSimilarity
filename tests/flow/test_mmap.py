from download_dataset import file_base_name, num_vectors # if download_dataset() is not commented it will run
import pickle
from VecSim import *
from common import create_hnsw_index, create_flat_index
import hnswlib
from numpy.testing import assert_allclose
import numpy as np

vectors_file_name = f"{file_base_name}_emb.pik"
with open(vectors_file_name,'rb') as f:
    unpickled_array = pickle.load(f)
    dim = unpickled_array.shape[1]
    print('Array shape: ' + str(unpickled_array.shape))
    print('Data type: '+str(type(unpickled_array)))


def test_sanity_vecsim_mmap():
    efRuntime = 10
    M = 16
    efConstruction = 100
    index = create_hnsw_index(
        dim,
        num_vectors,
        VecSimMetric_L2,
        VecSimType_FLOAT32,
        m=M,
        ef_construction=efConstruction,
        ef_runtime=efRuntime,
    )
    print("\ndisk hnsw index created")

    print("Create hnswlib index for sanity testing")
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_vectors, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    for i, vector in enumerate(unpickled_array[:num_vectors - 1]):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    print(f"disk hnsw index containts {index.index_size()} vectors")
    print(f"hnswlib index containts {p.get_current_count()} vectors")

    print("Testing knn")
    query_data = unpickled_array[-1]
    hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
    redis_labels, redis_distances = index.knn_query(query_data, 10)
    assert_allclose(hnswlib_labels, redis_labels, rtol=1e-5, atol=0)
    # print(f"redis labels = {redis_labels}, hnswlib labels = {hnswlib_labels}")
    assert_allclose(hnswlib_distances, redis_distances, rtol=1e-5, atol=0)
    # print(f"redis distances = {redis_distances}, hnswlib distances = {hnswlib_distances}")
    print("Testing knn ok")

def test_gt_vecsim_mmap():
    num_vectors = 1000
    efRuntime = 10
    M = 16
    efConstruction = 100
    metric = VecSimMetric_L2
    data_type = VecSimType_FLOAT32
    index = create_hnsw_index(
        dim=dim,
        num_elements=num_vectors,
        metric=metric,
        data_type=data_type,
        m=M,
        ef_construction=efConstruction,
        ef_runtime=efRuntime,
    )
    print("\ndisk hnsw index created")

    bf_index = create_flat_index(dim = dim, metric=metric, data_type=data_type)
    print("\ndisk bf_index index created")

    for i, vector in enumerate(unpickled_array[:num_vectors - 1]):
        index.add_vector(vector, i)
        bf_index.add_vector(vector, i)

    print(f"disk hnsw index contains {index.index_size()} vectors")
    print(f"disk bf index contains {bf_index.index_size()} vectors")

    print("Testing knn")
    query_data = unpickled_array[-1]
    flat_labels, flat_distances = bf_index.knn_query(query_data, k=10)
    hnsw_labels, hnsw_distances = index.knn_query(query_data, 10)

    fail = False
    res = np.allclose(flat_labels, hnsw_labels, rtol=1e-5, atol=0)
    if not res:
        print("Testing knn labels not ok")
        fail = True
    print(f"hnsw labels = {hnsw_labels}, flat labels = {flat_labels}")
    res = np.allclose(flat_distances, hnsw_distances, rtol=1e-5, atol=0)
    if not res:
        print("Testing knn dists not ok")
        fail = True
    print(f"hnsw distances = {hnsw_distances}, flat distances = {flat_distances}")
    assert not fail
# tiered_hnsw_params = create_tiered_hnsw_params(swap_job_threshold)

# tiered_index = Tiered_HNSWIndex(hnsw_params, tiered_hnsw_params, flat_buffer_size)
