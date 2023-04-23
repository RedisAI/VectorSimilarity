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
class DBPediaIndexCtx:
    def __init__(self, data_size = 0, M = 32, ef_c = 512, ef_r = 10):
        self.M = M
        self.efConstruction = ef_c
        self.efRuntime = ef_r 
        
        data = load_data("dbpedia-768")
        self.num_elements = data_size if data_size != 0 else data.shape[0]
        
        print(self.num_elements)
        self.data = data[:self.num_elements]
        self.dim = len(self.data[0])
        self.metric = VecSimMetric_Cosine
        self.type = VecSimType_FLOAT32
        
        self.hnsw_params = create_hnsw_params(self.dim, self.num_elements, self.metric, self.type, ef_c, M, ef_r)
        self.tiered_hnsw_params = TIERED_HNSWParams()
        self.tiered_hnsw_params.i = 0  
        
        self.tiered_index = TIERED_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params)
        
    def init_and_populate_flat_index(self):
        bfparams = BFParams()
        bfparams.initialCapacity = self.num_elements
        bfparams.dim =self.dim
        bfparams.type =self.type
        bfparams.metric =self.metric
        self.flat_index = BFIndex(bfparams)
        
        for i, vector in enumerate(self.data):
            self.flat_index.add_vector(vector, i)
        
        return self.flat_index
    
    def init_and_populate_hnsw_index(self):
        hnsw_index = HNSWIndex(self.hnsw_params)
        
        for i, vector in enumerate(self.data):
            hnsw_index.add_vector(vector, i)
        self.hnsw_index = hnsw_index
        return hnsw_index
    
    
        
def create_dbpedia():
    
    indices_ctx = DBPediaIndexCtx()
    index = indices_ctx.tiered_index
    num_elements = indices_ctx.num_elements 
    
    threads_num = index.get_threads_num()
    data = indices_ctx.data
    
    # Measure insertion to tiered index
    
    print(f"Insert {num_elements} vectors to tiered index")
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    index.wait_for_index()
    dur = time.time() - start
    
    print(f"Insert {num_elements} vectors to tiered index took {dur} s")
    
    # Measure total memory of the tiered index 
    print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} bytes")
    
    # Measure insertion to parallel index
    
    hnsw_parallel_index = HNSWIndex(indices_ctx.hnsw_params)
    print(f"Insert {num_elements} vectors to parallel hnsw index")
    start = time.time()
    hnsw_parallel_index.add_vector_parallel(data, np.array(range(num_elements)), num_threads=threads_num)
    dur = time.time() - start
    print(f"Insert {num_elements} vectors to parallel hnsw index took {dur} s")   
    
    
    hnsw_index = HNSWIndex(indices_ctx.hnsw_params)
    print(f"Insert {num_elements} vectors to hnsw index")
    start = time.time()
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
    dur = time.time() - start
    
    print(f"total memory of hnsw index = {hnsw_index.index_memory()} bytes")
    print(f"Insert {num_elements} vectors to hnsw index took {dur} s")   
    


def search_insert_dbpedia():
   
    indices_ctx = DBPediaIndexCtx()
    index = indices_ctx.tiered_index
    num_elements = indices_ctx.num_elements
    
    data = indices_ctx.data
    
    queries = load_queries("dbpedia-768")
    
    # Start background insertion the vectors into the tired index
    index_start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    
    correct = 0
    bf_total_time = 0
    hnsw_total_time = 0
    k = 10
    query_index = 0
    
    index.start_knn_log()
    hnsw_size_vs_query_time = []
    while index.hnsw_label_count() < num_elements:
        start = time.time()
        tiered_labels, tiered_distances = index.knn_query(queries[query_index], k)
        dur = time.time() - start
        hnsw_curr_size = index.get_curr_bf_size(mode = 'insert_and_knn')
        hnsw_size_vs_query_time.append((hnsw_curr_size, dur))
        print(hnsw_curr_size)
        time.sleep(1)
        query_index += 1
        print(f"search took {dur}")
    
    index_dur = time.time() - index_start
    print(f"search + indexing took  {index_dur} s")
    print(f"total memory of tiered index = {index.index_memory()} bytes")

     #Add all the vectors in the train set into the flat.
    bf_index = indices_ctx.init_and_populate_flat_index()

        
    num_queries = queries.shape[0]
    num_queries = 100
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

def sanity_test():
    num_elements = 10000
    k = num_elements
    
    indices_ctx = DBPediaIndexCtx(data_size = num_elements)
    index = indices_ctx.tiered_index    
    
    #add vectors to the tiered index one by one
    for i, vector in enumerate(indices_ctx.data):
        index.add_vector(vector, i)
        index.wait_for_index(5)
    
    print(f"done indexing, label count = {index.hnsw_label_count()}")
    
    # create hnsw inedx
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()
    
    queries = load_queries("dbpedia-768")
    #search knn in tiered
    tiered_labels, tiered_dist = index.knn_query(queries[0], k)
    #search knn in hnsw
    hnsw_labels, hnsw_dist = hnsw_index.knn_query(queries[0], k)
    
    #compare
    for i, hnsw_res_label in enumerate(hnsw_labels[0]):
        tiered_res_label = tiered_labels[0][i]
        if(hnsw_res_label != tiered_res_label):
            print(f"mismatched in pos {i}")
            print(f"hnsw_label is {hnsw_res_label}")
            print(f"tiered_label is {tiered_res_label}")
            
            hnsw_label_to_query_dist_hnsw = hnsw_index.get_distance_from(hnsw_res_label, queries[0])
            hnsw_label_to_query_dist_tiered = index.get_distance_from(hnsw_res_label, queries[0])
            print(f"distance from {hnsw_res_label} query in hnsw is {hnsw_label_to_query_dist_hnsw}")
            print(f"distance from {hnsw_res_label} query in tiered is {hnsw_label_to_query_dist_tiered}")
            hnsw_label_to_query_dist_hnsw = hnsw_index.get_distance_from(hnsw_res_label, queries[0])
            hnsw_label_to_query_dist_tiered = index.get_distance_from(hnsw_res_label, queries[0])
            print(f"distance from {hnsw_res_label} query in tiered is {hnsw_label_to_query_dist_tiered}")
            print(f"distance from {hnsw_res_label} query in hnsw is {hnsw_label_to_query_dist_hnsw}\n")
        
        #if the label is not the same
            #get the disteance with get_distance()

def test_main():

    print("Test creation")
    #create_dbpedia()
    
    ("Test search")
    search_insert_dbpedia()
   
    sanity_test()
    
#test_main()   


