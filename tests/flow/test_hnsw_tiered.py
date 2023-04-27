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

def create_tiered_hnsw_params(swap_job_threshold = 1024):
    tiered_hnsw_params = TieredHNSWParams()
    tiered_hnsw_params.swapJobThreshold  = swap_job_threshold
    return tiered_hnsw_params   

class DBPediaIndexCtx:
    def __init__(self, data_size = 0, M = 32, ef_c = 512, ef_r = 10, metric = VecSimMetric_Cosine, mode=CreationMode.ONLY_PARAMS):
        self.M = M
        self.efConstruction = ef_c
        self.efRuntime = ef_r 
        
        data = load_data("dbpedia-768")
        self.num_elements = data_size if data_size != 0 else data.shape[0]
        
        print(self.num_elements)
        self.data = data[:self.num_elements]
        self.dim = len(self.data[0])
        self.metric = metric
        self.type = VecSimType_FLOAT32
        
        self.hnsw_params = create_hnsw_params(self.dim, self.num_elements, self.metric, self.type, ef_c, M, ef_r)
        self.tiered_hnsw_params = create_tiered_hnsw_params()
        
        assert isinstance(mode, CreationMode)
        if mode == CreationMode.CREATE_TIERED_INDEX: 
            self.tiered_index = TIERED_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params)
        
    def create_tiered(self):
        return TIERED_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params)
        
    def init_and_populate_flat_index(self, data = None):
        bfparams = BFParams()
        bfparams.initialCapacity = self.num_elements
        bfparams.dim =self.dim
        bfparams.type =self.type
        bfparams.metric =self.metric
        self.flat_index = BFIndex(bfparams)
        
        if data is None:
            data = self.data
        for i, vector in enumerate(data):
            self.flat_index.add_vector(vector, i)
        
        return self.flat_index
    
    def init_and_populate_hnsw_index(self):
        hnsw_index = HNSWIndex(self.hnsw_params)
        
        for i, vector in enumerate(self.data):
            hnsw_index.add_vector(vector, i)
        self.hnsw_index = hnsw_index
        return hnsw_index
    
    
def create_dbpedia():
    
    indices_ctx = DBPediaIndexCtx(data_size=100000)
    num_elements = indices_ctx.num_elements 
    
    threads_num = TIEREDIndex.get_threads_num()
    print(f"thread num = {threads_num}")
    data = indices_ctx.data
    
    def create_tiered():
        index = indices_ctx.create_tiered()
        
        print(f"Insert {num_elements} vectors to tiered index")
        start = time.time()
        for i, vector in enumerate(data):
            index.add_vector(vector, i)
        print(f"current flat buffer size is {index.get_curr_bf_size()}, wait for index\n")
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
   # create_hnsw()

def search_insert_dbpedia():
    indices_ctx = DBPediaIndexCtx(data_size=100000, mode=CreationMode.CREATE_TIERED_INDEX)
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
    index.wait_for_index(0.5)
    index_dur = time.time() - index_start
    print(f"search + indexing took {index_dur} s")
    print(f"total memory of tiered index = {index.index_memory()} bytes")
        
    
    # Measure recall
    recall = float(correct)/(k*searches_number)
    print("Average recall is:", recall)
    print("tiered query per seconds: ", searches_number/total_tiered_search_time)   

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
        index.wait_for_index(1)
    
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
    


def test_main():
    

    print("Test creation")
    create_dbpedia()
    
    print("Test search and insert in parallel")
   # search_insert_dbpedia()
    
    print("Sanity test")
  #  sanity_test()
    


# only search 
# insert multi
# search insetrt multi
# batch?

# delete
def recall_after_deletion():
    num_elements = 10000
    
    indices_ctx = DBPediaIndexCtx(data_size = num_elements, mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index
    data = indices_ctx.data
    
    # Populate tiered index 
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    print(f"current flat buffer size is {index.get_curr_bf_size()}, wait for index\n")
    index.wait_for_index()
    
    #delete half of the index
    start_delete = time.time()
    for i in range(0, num_elements, 2):
        index.delete_vector(i)
    # wait for all repair jobs to be done
    index.wait_for_index(5)
    deletion_time = time.time() - start_delete
    
    assert index.hnsw_label_count() == (num_elements / 2)
    
    print(f"Delete {num_elements / 2} vectors from tiered hnsw index took {deletion_time} s")   
    
    # init bf only with the vectors we didn't delete
    indices_ctx.init_and_populate_flat_index(data = data[:num_elements:2])
    # perfom 10 querires
        #compare recall with bf index 
    

#delete with search
#delete with insert    
