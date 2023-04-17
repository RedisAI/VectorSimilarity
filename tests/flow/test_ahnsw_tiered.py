# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import concurrent
import math
import multiprocessing
import os
import time
from common import *
import hnswlib

def test_create_new_index():
    print("hello")
    #create index
    dim = 16
    num_elements = 100000
    space = 'cosine'
    M = 16
    efConstruction = 100
    efRuntime = 10

    hnsw_params = create_hnsw_params(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    tiered_hnsw_params = TIERED_HNSWParams()
    tiered_hnsw_params.i = 0
    index = TIERED_HNSWIndex(hnsw_params, tiered_hnsw_params)
    x = input("meow")
    print(index.index_size())
    print("meow")
    
    hnsw_index =  create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    data = np.float32(np.random.random((num_elements, dim)))
    print(f"label count = {index.hnsw_label_count()}")
    
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    index.wait_for_index()
    dur = time.time() - start
    
    
    print(f"indexing tiered took{dur}")
    print(f"label count = {index.hnsw_label_count()}")
    start = time.time()
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
    dur = time.time() - start
    
    print(index.index_size())
    print(f"indexing hnsw took{dur}")
    #add
    #search
    #compare with hnsw

