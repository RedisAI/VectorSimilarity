
# use to disable automatic dataset download in download_dataset.py
import os
os.environ['DOWNLOAD_MULTILANG_DATASET'] = 'false'
os.environ['VERIFY_MULTILANG_DATASET'] = 'false'

from download_dataset import file_base_name, num_vectors_train, num_vectors_test # if download_dataset() is not commented it will run
import pickle
from VecSim import *
from common import create_hnsw_index, create_flat_index
import hnswlib
from numpy.testing import assert_allclose
import numpy as np
import time

# Globals
RUN_SANITY = True
RUN_BM = False

vectors_file_name = f"{file_base_name}_emb.pik"
with open(vectors_file_name,'rb') as f:
    print(f"loading vectors files from {vectors_file_name}")
    unpickled_array = pickle.load(f)
    dim = unpickled_array.shape[1]
    print('Array shape: ' + str(unpickled_array.shape))
    print('Data type: '+str(type(unpickled_array)))

def bm_build(M, efC):
    index = create_hnsw_index(
        dim,
        num_vectors_train,
        VecSimMetric_L2,
        VecSimType_FLOAT32,
        m=M,
        ef_construction=efC,
    )

    # index.disable_logs()
    print("\nBuilding index with params: ", f"M: {M}, efC{efC}\n")
    start_time = time.time()  # Start timing
    for i, vector in enumerate(unpickled_array[:num_vectors_train]):
        index.add_vector(vector, i)
    end_time = time.time()  # End timing
    print('\nBuilding time: ',f"T{end_time - start_time:.4f} seconds\n")

    # Sanity checks
    print(f"disk hnsw index contains {index.index_size()} vectors")

    random_query_index = np.random.randint(0, num_vectors_train)
    query_data = unpickled_array[random_query_index]
    labels, distances = index.knn_query(query_data, k=1)
    print(f"knn query result: {labels} {distances}")
    assert labels[0][0] == random_query_index
    assert distances[0][0] == float(0)

def bm():
    # Build params
    Ms_efC = [(60, 75), (120, 150), (150, 150), (200, 120), (200, 150)]

    bm_build(Ms_efC[0][0], Ms_efC[0][1])
    # query params
    Ks = [1, 10, 100]
    factors = [2, 10, 20, 100, 200]
    Ks_efRList = []
    max_efR = 200
    for k in Ks:
        efRs = []
        for factor in factors:
            if k * factor < max_efR:
                efRs.append(k * factor)
        Ks_efRList.append((k, efRs))


def sanity_vecsim_mmap():
    efRuntime = 10
    M = 16
    efConstruction = 100

    num_vectors = min(1000, num_vectors_train - 1)

    index = create_hnsw_index(
        dim,
        num_vectors_train,
        VecSimMetric_L2,
        VecSimType_FLOAT32,
        m=M,
        ef_construction=efConstruction,
        ef_runtime=efRuntime,
    )
    print("\ndisk hnsw index created")

    print("Create hnswlib index for sanity testing")
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_vectors_train, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    for i, vector in enumerate(unpickled_array[:num_vectors]):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    print(f"disk hnsw index containts {index.index_size()} vectors")
    print(f"hnswlib index containts {p.get_current_count()} vectors")

    print("Testing knn")
    query_data = unpickled_array[num_vectors]
    hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
    redis_labels, redis_distances = index.knn_query(query_data, 10)
    assert_allclose(hnswlib_labels, redis_labels, rtol=1e-5, atol=0)
    # print(f"redis labels = {redis_labels}, hnswlib labels = {hnswlib_labels}")
    assert_allclose(hnswlib_distances, redis_distances, rtol=1e-5, atol=0)
    # print(f"redis distances = {redis_distances}, hnswlib distances = {hnswlib_distances}")
    print("Testing knn ok")

def gt_vecsim_mmap():
    num_vectors = min(1000, num_vectors_train - 1)
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

    for i, vector in enumerate(unpickled_array[:num_vectors]):
        index.add_vector(vector, i)
        bf_index.add_vector(vector, i)

    print(f"disk hnsw index contains {index.index_size()} vectors")
    print(f"disk bf index contains {bf_index.index_size()} vectors")

    print("Testing knn")
    query_data = unpickled_array[num_vectors]
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

def test_sanity():
    if (RUN_SANITY):
        sanity_vecsim_mmap()
        gt_vecsim_mmap()

def test_bm():
    if (RUN_BM):
        bm()
