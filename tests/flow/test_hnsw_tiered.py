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
import h5py
import pickle


def load_data(dataset_name):
    data = 0

    np_data_file_path = os.path.join('np_train_%s.npy' % dataset_name)

    try:
        data = np.load(np_data_file_path, allow_pickle = True)
        print(f"yay! loaded ")
    except:
        hdf5_filename = os.path.join('%s.hdf5' % dataset_name)
        dataset = h5py.File(hdf5_filename, 'r')
        data = np.array(dataset['train'])
        np.save(np_data_file_path, data)
        print(f"yay! generated")
    
    return data    

def load_queries(dataset_name):
    queries = 0
    np_test_file_path = os.path.join('np_test_%s.npy' % dataset_name)

    try:
        queries = np.load(np_test_file_path, allow_pickle = True)
        print(f"yay! loaded ")
    except:
        hdf5_filename = os.path.join('%s.hdf5' % dataset_name)
        dataset = h5py.File(hdf5_filename, 'r')
        queries = np.array(dataset['test'])
        np.save(np_test_file_path, queries)
        print(f"yay! generated ")
    
    return queries    
  

def create_dbpedia(hnsw_params, tiered_hnsw_params, data):
    
    index = TIERED_HNSWIndex(hnsw_params, tiered_hnsw_params)
    
    threads_num = index.get_threads_num()
    num_elements = data.shape[0]
    
    print(f"Insert {num_elements} vectors to tiered index")
    dur = 0
    for i, vector in enumerate(data):
        start = time.time()
        index.add_vector(vector, i)
        dur += time.time() - start
        if(i % 10000 == 0):
            print(f"inserted {i} elements to the flat buffer\n")
    index.wait_for_index()
    
    print(f"Insert {num_elements} vectors to tiered index took {dur} s")
    
    hnsw_index = HNSWIndex(hnsw_params)
    print(f"Insert {num_elements} vectors to hnsw index")
    start = time.time()
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
    dur = time.time() - start
    
    print(f"Insert {num_elements} vectors to hnsw index took {dur} s")   

    hnsw_parallel_index = HNSWIndex(hnsw_params)
    print(f"Insert {num_elements} vectors to parallel hnsw index")
    start = time.time()
    hnsw_parallel_index.add_vector_parallel(data, np.array(range(num_elements)), num_threads=threads_num)
    dur = time.time() - start
    
    print(f"Insert {num_elements} vectors to parallel hnsw index took {dur} s")   

def search_insert_dbpedia(hnsw_params, tiered_hnsw_params, bf_index, data):
    index = TIERED_HNSWIndex(hnsw_params, tiered_hnsw_params)

    num_elements = data.shape[0]

    test = 100000
    data = data[:test]
     #Add all the vectors in the train set into the flat.
    for i, vector in enumerate(data):
        bf_index.add_vector(vector, i)
    
    queries = load_queries("dbpedia-768")
    
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    dur = time.time() - start
    
    correct = 0
    bf_total_time = 0
    hnsw_total_time = 0
    num_queries = 1
    k = 10
    while index.hnsw_label_count() < num_elements:
        start = time.time()
        tiered_labels, tiered_distances = index.knn_query(queries[0], k)
        dur = time.time() - start
        time.sleep(0.5)
        print(f"search took{dur}")
        
    # Measure recall and times
    for target_vector in queries[:num_queries]:
        start = time.time()
        tiered_labels, tiered_distances = index.knn_query(target_vector, k)
        hnsw_total_time += (time.time() - start)
        start = time.time()
        bf_labels, bf_distances = bf_index.knn_query(target_vector, k)
        bf_total_time += (time.time() - start)
        correct += len(np.intersect1d(tiered_labels[0], bf_labels[0]))    
    
     # Measure recall
    recall = float(correct)/(k*num_queries)
    print("Average recall is:", recall)
    print("BF query per seconds: ", num_queries/bf_total_time)
    print("tiered query per seconds: ", num_queries/hnsw_total_time)   

def test_insert_search():
    dim = 16
    num_elements = 10000
    M = 16
    efConstruction = 100
    efRuntime = 10

    hnsw_params = create_hnsw_params(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    tiered_hnsw_params = TIERED_HNSWParams()
    tiered_hnsw_params.i = 0
    index = TIERED_HNSWIndex(hnsw_params, tiered_hnsw_params)
    threads_num = index.get_threads_num()
    
    #insert vector
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    dur = time.time() - start
    
    #query
    #get label count
    #measure search time
        
def test_create_new_index():
    dim = 16
    num_elements = 10000
    M = 16
    efConstruction = 100
    efRuntime = 10

    hnsw_params = create_hnsw_params(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    tiered_hnsw_params = TIERED_HNSWParams()
    tiered_hnsw_params.i = 0
    index = TIERED_HNSWIndex(hnsw_params, tiered_hnsw_params)
    threads_num = index.get_threads_num()
    
    hnsw_index =  create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    parallel_index =  create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
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
    parallel_index.add_vector_parallel(data, np.array(range(num_elements)), num_threads=threads_num)

    dur = time.time() - start
    print(f"parallel label count = {parallel_index.index_size()}")
    
    
    print(f"indexing parallel_index took{dur}")
    start = time.time()
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
    dur = time.time() - start
    
    print(index.index_size())
    print(f"indexing hnsw took{dur}")
    #add
    #search
    k = 10
    # query_data = np.float32(np.random.random((1, dim)))

    
    # tiered_labels, tiered_distances = index.knn_query(query_data, k)
    # hnsw_labels, hnsw_distances = hnsw_index.knn_query(query_data, k)
    # assert_allclose(tiered_labels, hnsw_labels, rtol=1e-5, atol=0)
    # assert_allclose(tiered_distances, hnsw_distances, rtol=1e-5, atol=0)
    # print(f"tiered labels = {tiered_labels}")
    # print(f"hnsw labels = {hnsw_labels}")
    # print(f"tiered dist = {tiered_distances}")
    # print(f"hnsw dist = {hnsw_distances}")
    
    num_queries = 1000
    query_data = np.float32(np.random.random((num_queries, dim)))  
    start = time.time()
    for i, query in enumerate(query_data):
        hnsw_labels, hnsw_distances = hnsw_index.knn_query(query, k)
    total_search_time_hnsw = time.time() - start
    
    print(f"search hnsw took {total_search_time_hnsw}")
    
    start = time.time()
    parallel_labels, parallel_distances = hnsw_index.knn_parallel(query_data, k, num_threads=threads_num)
    total_search_time_parallel = time.time() - start
    
    print(f"search in parallel took {total_search_time_parallel}")
    
    
    start = time.time()
    for i, query in enumerate(query_data):
        tiered_labels, tiered_distances = index.knn_query(query, k)
    total_search_time_tiered = time.time() - start
    
    print(f"search tiered took {total_search_time_tiered}")
    
    # assert_allclose(tiered_labels, hnsw_labels, rtol=1e-5, atol=0)
    # assert_allclose(parallel_labels, hnsw_labels, rtol=1e-5, atol=0)
    
    # assert_allclose(tiered_distances, hnsw_distances, rtol=1e-5, atol=0)
    # assert_allclose(parallel_distances, hnsw_distances, rtol=1e-5, atol=0)
    
    # print(f"tiered labels = {tiered_labels}")
    # print(f"hnsw labels = {hnsw_labels}")
    # print(f"parallel labels = {parallel_labels}")
    # print(f"tiered dist = {tiered_distances}")
    # print(f"hnsw dist = {hnsw_distances}")
    # print(f"hnsw dist = {parallel_distances}")
    
    #compare with hnsw
def test_serach():
    dim = 16
    num_elements = 100000
    M = 16
    efConstruction = 100
    efRuntime = 10

    hnsw_params = create_hnsw_params(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    tiered_hnsw_params = TIERED_HNSWParams()
    tiered_hnsw_params.i = 0
    index = TIERED_HNSWIndex(hnsw_params, tiered_hnsw_params)
    threads_num = index.get_threads_num()
    
    hnsw_index =  create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    hnsw_index2 =  create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    parallel_index =  create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    data = np.float32(np.random.random((num_elements, dim)))
    print(f"label count = {index.hnsw_label_count()}")
    
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    #    index.wait_for_index()
    dur = time.time() - start
    
    k = 10
    num_queries = 100
    query_data = np.float32(np.random.random((1, dim)))
    
    while index.hnsw_label_count() < num_elements:
        start = time.time()
        tiered_labels, tiered_distances = index.knn_query(query_data, k)
        dur = time.time() - start
        time.sleep(0.05)
        print(f"isearch took{dur}")
        
    start = time.time()
    tiered_labels, tiered_distances = index.knn_query(query_data, k)
    dur = time.time() - start
    print(f"isearch took{dur}")   
    
    
    print(f"label count = {index.hnsw_label_count()}")
    
    start = time.time()
    parallel_index.add_vector_parallel(data, np.array(range(num_elements)), num_threads=threads_num)

    dur = time.time() - start
    print(f"parallel label count = {parallel_index.index_size()}")
    
    
    print(f"indexing parallel_index took{dur}")
    start = time.time()
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
    dur = time.time() - start
    
    print(hnsw_index.index_size())
    print(f"indexing hnsw took{dur}")

    #create tiered
    #insert one by one (call wait for index after each insertion)
    
    #create hnsw
    #create parallel
    
  
    
    hnsw_labels, hnsw_distances = hnsw_index.knn_query(query_data, k)
    hnsw_labels2, hnsw_distances2 = hnsw_index2.knn_query(query_data, k)
    for i, label in enumerate(hnsw_labels[0]):
      if label != tiered_labels[0][i]:
        print(f"pos = {i}")
        print(f"label hnsw= {label}")
        print(f"dist hnsw= {hnsw_distances[0][i]}")
        
        print(f"label tiers= {tiered_labels[0][i]}")
        print(f"label tiers= {tiered_distances[0][i]}")
        
    print(np.array_equal(hnsw_labels[0], tiered_labels[0]))
    # assert_allclose(parallel_labels, hnsw_labels, rtol=1e-5, atol=0)
    
    # assert_allclose(tiered_distances, hnsw_distances, rtol=1e-5, atol=0)
    # assert_allclose(parallel_distances, hnsw_distances, rtol=1e-5, atol=0)
#כמה וקטורים שנארו לי לאנדקס כפונקציה של כמה זמן לקח השאילתה
 
def test_main():
    data = load_data("dbpedia-768")
    
    M = 64
    efConstruction = 512
    efRuntime = 10 
    dim = len(data[0])
    print(f"dim = {dim}")
    num_elements = data.shape[0]
    print(f"num_elements = {num_elements}")
    
    metric = VecSimMetric_Cosine
    hnsw_params = create_hnsw_params(dim, num_elements, metric, VecSimType_FLOAT32, efConstruction, M, efRuntime)
    tiered_hnsw_params = TIERED_HNSWParams()
    tiered_hnsw_params.i = 0        
    
    print("Test creation")
    create_dbpedia(hnsw_params,tiered_hnsw_params, data)
    
    ("Test search")
    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
    bfparams.dim = dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = metric
    bf_index = BFIndex(bfparams)
   # search_insert_dbpedia(hnsw_params,tiered_hnsw_params, bf_index, data)
    
test_main()   
