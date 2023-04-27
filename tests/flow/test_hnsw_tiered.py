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
from urllib.request import urlretrieve
import pickle
from enum import Enum

class CreationMode(Enum):
    ONLY_PARAMS = 1
    CREATE_TIERED_INDEX = 2

def download(src, dst):
    if not os.path.exists(dst):
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

# Download dataset from s3, save the file locally
def get_data_set(dataset_name):
    hdf5_filename = os.path.join('%s.hdf5' % dataset_name)
    url = 'https://s3.amazonaws.com/benchmarks.redislabs/vecsim/dbpedia/dbpedia-768.hdf5'
    download(url, hdf5_filename)
    return h5py.File(hdf5_filename, 'r')

def load_data(dataset_name):
    data = 0

    np_data_file_path = os.path.join('np_train_%s.npy' % dataset_name)

    try:
        data = np.load(np_data_file_path, allow_pickle = True)
        print(f"yay! loaded ")
    except:
        dataset = get_data_set(dataset_name)
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

# swap_job_threshold = 0 means use the default swap_job_threshold defined in hnsw_tiered.h
def create_tiered_hnsw_params(swap_job_threshold = 0):
    tiered_hnsw_params = TieredHNSWParams()
    tiered_hnsw_params.swapJobThreshold  = swap_job_threshold
    return tiered_hnsw_params   

class DBPediaIndexCtx:
    def __init__(self, data_size = 0, M = 32, ef_c = 512, ef_r = 10, metric = VecSimMetric_Cosine, is_multi = False, data_type = VecSimType_FLOAT32, swap_job_threshold = 0, mode=CreationMode.ONLY_PARAMS):
        self.M = M
        self.efConstruction = ef_c
        self.efRuntime = ef_r 
        
        data = load_data("dbpedia-768")
        self.num_elements = data_size if data_size != 0 else data.shape[0]
        
        self.data = data[:self.num_elements]
        self.dim = len(self.data[0])
        self.metric = metric
        self.type = data_type
        self.is_multi = is_multi
        
        self.hnsw_params = create_hnsw_params(dim=self.dim, 
                                              num_elements=self.num_elements, 
                                              metric=self.metric,
                                              data_type=self.type,
                                              ef_construction=ef_c,
                                              m=M,
                                              ef_runtime=ef_r,
                                              is_multi=self.is_multi)
        self.tiered_hnsw_params = create_tiered_hnsw_params(swap_job_threshold)
        
        assert isinstance(mode, CreationMode)
        if mode == CreationMode.CREATE_TIERED_INDEX: 
            self.tiered_index = TIERED_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params)
        
    def create_tiered(self):
        return TIERED_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params)
    
    def set_num_vectors_per_label(self, num_per_label = 1):
        self.num_per_label = num_per_label
        
    def init_and_populate_flat_index(self):
        bfparams = BFParams()
        bfparams.initialCapacity = self.num_elements
        bfparams.dim =self.dim
        bfparams.type =self.type
        bfparams.metric =self.metric
        bfparams.multi = self.is_multi
        self.flat_index = BFIndex(bfparams)
        
        for i, vector in enumerate(self.data):
            for _ in range(self.num_per_label):
                self.flat_index.add_vector(vector, i)
        
        return self.flat_index
    
    def init_and_populate_hnsw_index(self):
        hnsw_index = HNSWIndex(self.hnsw_params)
        
        for i, vector in enumerate(self.data):
            hnsw_index.add_vector(vector, i)
        self.hnsw_index = hnsw_index
        return hnsw_index
    
    
def create_dbpedia():
    num_elements = 100000
    
    indices_ctx = DBPediaIndexCtx(data_size = num_elements)
    
    threads_num = TIEREDIndex.get_threads_num()
    print(f"thread num = {threads_num}")
    data = indices_ctx.data
    data = np.float32(np.random.random((num_elements, indices_ctx.dim)))
    
    
    def create_tiered():
        index = indices_ctx.create_tiered()
        
        print(f"Insert {num_elements} vectors to tiered index")
        start = time.time()
        for i, vector in enumerate(data):
            index.add_vector(vector, i)
        bf_dur = time.time() - start
        
        print(f"current flat buffer size is {index.get_curr_bf_size()}, took{bf_dur} wait for index\n")
        index.wait_for_index()
        dur = time.time() - start
        
        assert index.hnsw_label_count() == num_elements
        
        # Measure insertion to tiered index
        
        print(f"Insert {num_elements} vectors to tiered index took {dur} s")
        
        # Measure total memory of the tiered index 
        print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
    
    print(f"Start tiered hnsw creation")
    create_tiered()
    
    def create_parallel():
        # Measure insertion to parallel index
        hnsw_parallel_index = HNSWIndex(indices_ctx.hnsw_params)
        print(f"Insert {num_elements} vectors to parallel hnsw index")
        start = time.time()
        hnsw_parallel_index.add_vector_parallel(data, np.array(range(num_elements)), num_threads=threads_num)
        dur = time.time() - start
        print(f"total memory of hnsw index = {hnsw_parallel_index.index_memory()/pow(10,9)} GB")
        print(f"Insert {num_elements} vectors to parallel hnsw index took {dur} s")   
    
    print(f"Start parallel hnsw creation")
    create_parallel()
    
    def create_hnsw():
        hnsw_index = HNSWIndex(indices_ctx.hnsw_params)
        print(f"Insert {num_elements} vectors to hnsw index")
        start = time.time()
        for i, vector in enumerate(data):
            hnsw_index.add_vector(vector, i)
        dur = time.time() - start
        
        print(f"total memory of hnsw index = {hnsw_index.index_memory()/pow(10,9)} GB")
        print(f"Insert {num_elements} vectors to hnsw index took {dur} s")   
    
    print(f"Start sync hnsw creation")
    create_hnsw()

def search_insert_dbpedia(is_multi: bool, num_per_label = 1):
    indices_ctx = DBPediaIndexCtx(mode=CreationMode.CREATE_TIERED_INDEX, is_multi=is_multi)
    index = indices_ctx.tiered_index
    
    num_elements = indices_ctx.num_elements
    data = indices_ctx.data
    
    queries = load_queries("dbpedia-768")
    num_queries = queries.shape[0]
    
    indices_ctx.set_num_vectors_per_label(num_per_label)
    
    # Add vectors into the flat index.
    bf_index = indices_ctx.init_and_populate_flat_index()
    
    # Start background insertion in the tiered index
    index_start = time.time()
    for i, vector in enumerate(data):
        for _ in range(num_per_label):
            index.add_vector(vector, i)
    
    correct = 0
    k = 10
    query_index = 0
    searches_number = 0
    
    # config the index log to knn mode to get access to the bf index size during the query.
    index.start_knn_log()
    
    # run knn query every 1 s 
    total_tiered_search_time = 0
    while index.hnsw_label_count() < num_elements:
        query_start = time.time()
        tiered_labels, _ = index.knn_query(queries[query_index], k)
        query_dur = time.time() - query_start
        total_tiered_search_time += query_dur
        # for each run get the current hnsw size and the query time
        bf_curr_size = index.get_curr_bf_size(mode = 'insert_and_knn')
        print(f"query time = {query_dur}")
        print(f"bf size = {bf_curr_size}")
        
        # run the query also in the bf index to get the grund truth results
        bf_labels, _ = bf_index.knn_query(queries[query_index], k)
        correct += len(np.intersect1d(tiered_labels[0], bf_labels[0]))    
        time.sleep(10)
        query_index = min(query_index + 1, num_queries - 1)
        searches_number += 1
    
    # hnsw labels count updates before addVector returns, wait for the queue to be empty
    index.wait_for_index(1)
    index_dur = time.time() - index_start
    print(f"search + indexing took {index_dur} s")
    print(f"total memory of tiered index = {index.index_memory()} bytes")
        
    
    # Measure recall
    recall = float(correct)/(k*searches_number)
    print("Average recall is:", recall)
    print("tiered query per seconds: ", searches_number/total_tiered_search_time)   

def test_search_insert_dbpedia_single_index():
    search_insert_dbpedia(is_multi = False)
    
# def test_search_insert_dbpedia_multi_index():
#     search_insert_dbpedia(is_multi = True)
    
# In this test we insert the vectors one by one to the tiered index (call wait_for_index after each add vector)
# We expect to get the same index as if we were inserting the vector to the sync hnsw index.
# To check that, we perform a knn query with k = vectors number and compare the results' labels
# to pass the test all the labels and distances should be the same.
def sanity_test():
    num_elements = 10000
    k = num_elements
    
    indices_ctx = DBPediaIndexCtx(data_size = num_elements, mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index    
    
    #add vectors to the tiered index one by one
    for i, vector in enumerate(indices_ctx.data):
        index.add_vector(vector, i)
        index.wait_for_index()
    
    print(f"done indexing, label count = {index.hnsw_label_count()}")
    
    # create hnsw inedx
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()
    
    queries = load_queries("dbpedia-768")
    #search knn in tiered
    tiered_labels, tiered_dist = index.knn_query(queries[0], k)
    #search knn in hnsw
    hnsw_labels, hnsw_dist = hnsw_index.knn_query(queries[0], k)
    
    #compare
    has_diff = False
    for i, hnsw_res_label in enumerate(hnsw_labels[0]):
        if hnsw_res_label != tiered_labels[0][i]:
            has_diff = True
            print(f"hnsw label = {hnsw_res_label}, tiered label = {tiered_labels[0][i]}")
            print(f"hnsw dist = {hnsw_dist[0][i]}, tiered dist = {tiered_dist[0][i]}")

    assert has_diff == False
    print(f"cool")
    

def recall_after_deletion():
    num_elements = 10000
    
    indices_ctx = DBPediaIndexCtx(data_size = num_elements, mode=CreationMode.CREATE_TIERED_INDEX, ef_r=20)
    index = indices_ctx.tiered_index
    data = indices_ctx.data
    
    # Populate tiered index 
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))
    print(f"current flat buffer size is {index.get_curr_bf_size()}, wait for index\n")
    index.wait_for_index()
    
    #create and populate hnsw index
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()
    
    #delete half of the index
    start_delete = time.time()
    for i in range(0, num_elements, 2):
        index.delete_vector(i)
    # wait for all repair jobs to be done
    index.wait_for_index(5)
    deletion_time = time.time() - start_delete
    vectors = [vectors[i] for i in range(1, num_elements, 2)]
    assert index.hnsw_label_count() == (num_elements / 2)
    print(f"Delete {num_elements / 2} vectors from tiered hnsw index took {deletion_time} s")   
    
    # measure also hnsw time
    start_delete = time.time()
    for i in range(0, num_elements, 2):
        hnsw_index.delete_vector(i)
    deletion_time = time.time() - start_delete
    
    assert hnsw_index.index_size() == (num_elements / 2)
    
    print(f"Delete {num_elements / 2} vectors from hnsw index took {deletion_time} s")   
    
    # perfom querires
    num_queries = 10
    queries = load_queries("dbpedia-768")[:num_queries]
    
    k = 10
    correct_tiered = 0
    correct_hnsw = 0
    print(f"queries num = {queries.shape[0]}")
    
    # calculate correct vectors for each recall  
    # We don't expect hnsw and tiered hnsw results to be identical due to the parallel insertion.
    def calculate_correct(index_lables, keys):
        correct = 0;
        for label in index_lables[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct += 1
                    break 
        return correct
    
    index.start_knn_log()
    for target_vector in queries:
        tiered_labels, _ = index.knn_query(target_vector, k)
        hnsw_labels, _ = hnsw_index.knn_query(target_vector, k)
        
        # sort distances of every vector from the target vector and get actual k nearest vectors

        dists = [(spatial.distance.cosine(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        keys = [key for _, key in dists[:k]]
        correct_tiered += calculate_correct(tiered_labels, keys)
        correct_hnsw += calculate_correct(hnsw_labels, keys)

    # Measure recall
    recall_tiered = float(correct_tiered) / (k * num_queries)
    recall_hnsw = float(correct_hnsw) / (k * num_queries)
    print("\nhsnw tiered recall is: \n", recall_tiered)
    print("\nhsnw recall is: \n", recall_hnsw)
    assert (recall_tiered > 0.9)
    assert (recall_hnsw > 0.9)

def create_multi_index():
    indices_ctx = DBPediaIndexCtx(mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index
    
    num_elements = indices_ctx.num_elements
    data = indices_ctx.data
    
    queries = load_queries("dbpedia-768")
    num_queries = queries.shape[0]
    
    # Add vectors into the flat index.
    bf_index = indices_ctx.init_and_populate_flat_index()
    
    # Start background insertion in the tiered index
    index_start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    
    correct = 0
    k = 10
    query_index = 0
    searches_number = 0
    
    # config the index log to knn mode to get access to the bf index size during the query.
    index.start_knn_log()
    
    # run knn query every 1 s 
    total_tiered_search_time = 0
    while index.hnsw_label_count() < num_elements:
        query_start = time.time()
        tiered_labels, _ = index.knn_query(queries[query_index], k)
        query_dur = time.time() - query_start
        total_tiered_search_time += query_dur
        # for each run get the current hnsw size and the query time
        bf_curr_size = index.get_curr_bf_size(mode = 'insert_and_knn')
        print(f"query time = {query_dur}")
        print(f"bf size = {bf_curr_size}")
        
        # run the query also in the bf index to get the grund truth results
        bf_labels, _ = bf_index.knn_query(queries[query_index], k)
        correct += len(np.intersect1d(tiered_labels[0], bf_labels[0]))    
        time.sleep(10)
        query_index = min(query_index + 1, num_queries - 1)
        searches_number += 1
    
    # hnsw labels count updates before addVector returns, wait for the queue to be empty
    index.wait_for_index(1)
    index_dur = time.time() - index_start
    print(f"search + indexing took {index_dur} s")
    print(f"total memory of tiered index = {index.index_memory()} bytes")
        
    
    # Measure recall
    recall = float(correct)/(k*searches_number)
    print("Average recall is:", recall)
    print("tiered query per seconds: ", searches_number/total_tiered_search_time)   
    
    
def test_main():
    print("Test creation")
    create_dbpedia()
    
    print("Test search and insert in parallel")
    #search_insert_dbpedia()
    
    print("Sanity test")
   # sanity_test()
  
    print("recall after delete test")
    #recall_after_deletion()
    
    print("delete in parallel with search ")
   # delete_search()
  
    
 



# only search 
# insert multi
# search insetrt multi
# batch?

# delete search parallel
