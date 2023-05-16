# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import concurrent
import math
import multiprocessing
import os
import time
from common import *
import h5py
from urllib.request import urlretrieve
import pickle
from enum import Enum

from random import choice

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
    print(f"download {hdf5_filename}")
    url = 'https://s3.amazonaws.com/benchmarks.redislabs/vecsim/dbpedia/dbpedia-768.hdf5'
    download(url, hdf5_filename)
    return h5py.File(hdf5_filename, 'r')

def load_data(dataset_name):
    data = 0

    np_data_file_path = os.path.join('np_train_%s.npy' % dataset_name)

    try:
        print(f"try to load {np_data_file_path}")
        data = np.load(np_data_file_path, allow_pickle = True)
        print(f"yay! loaded ")
    except:
        print(f"failed to load {np_data_file_path}")
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
    def __init__(self, data_size = 0, initialCap = 0, M = 32, ef_c = 512, ef_r = 10, metric = VecSimMetric_Cosine, is_multi = False, data_type = VecSimType_FLOAT32, swap_job_threshold = 0, mode=CreationMode.ONLY_PARAMS):
        self.M = M
        self.efConstruction = ef_c
        self.efRuntime = ef_r 
        
        data = load_data("dbpedia-768")
        self.num_elements = data_size if data_size != 0 else data.shape[0]
        #self.initialCap = initialCap if initialCap != 0 else 2 * self.num_elements
        self.initialCap = initialCap if initialCap != 0 else self.num_elements
        
        self.data = data[:self.num_elements]
        self.dim = len(self.data[0])
        self.metric = metric
        self.data_type = data_type
        self.is_multi = is_multi
        
        self.hnsw_params = create_hnsw_params(dim=self.dim, 
                                              num_elements=self.initialCap, 
                                              metric=self.metric,
                                              data_type=self.data_type,
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
            
    def create_hnsw(self):
        return HNSWIndex(self.hnsw_params)
        
    def init_and_populate_flat_index(self):
        bfparams = BFParams()
        bfparams.initialCapacity = self.num_elements
        bfparams.dim =self.dim
        bfparams.type =self.data_type
        bfparams.metric =self.metric
        bfparams.multi = self.is_multi
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
    
    def populate_index(self, index):
        start = time.time()
        duration = 0
        for label, vector in enumerate(self.data):
            start_add = time.time()
            index.add_vector(vector, label)
            duration += time.time() - start_add
            if label % 1000 == 0:
                print(f"time passes= {duration}")
        end = time.time()
        return (start, duration, end)
    
    def generate_random_vectors(self, num_vectors):
        vectors = 0
        np_file_path = os.path.join(f'np_{num_vectors}vec_dim{self.dim}.npy')

        try:
            vectors = np.load(np_file_path, allow_pickle = True)
            print(f"yay! loaded ")
        except:
            rng = np.random.default_rng(seed=47)
            vectors = np.float32(rng.random((num_vectors, self.dim)))
            np.save(np_file_path, vectors)
            print(f"yay! generated ")
        
        return vectors     
        
    def insert_in_batch(self, index, data, data_first_idx, batch_size, first_label):
        duration = 0
        data_last_idx = data_first_idx + batch_size
        for i, vector in enumerate(data[data_first_idx:data_last_idx]):
            label = i + first_label
            start_add = time.time()
            index.add_vector(vector, label)
            duration += time.time() - start_add
        end = time.time()
        return (duration, end)
   
    def generate_queries(self, num_queries):
        self.rng = np.random.default_rng(seed=47)
        
        queries = self.rng.random((num_queries, self.dim)) 
        return np.float32(queries) if self.data_type == VecSimType_FLOAT32 else queries
    
    def generate_query_from_ds(self):
        return choice(self.data)
        
        
def create_dbpedia():
    indices_ctx = DBPediaIndexCtx(data_size= 1000000)
    
    threads_num = TIEREDIndex.get_threads_num()
    print(f"thread num = {threads_num}")
    data = indices_ctx.data
    num_elements = indices_ctx.num_elements
    def create_parallel():
        index = indices_ctx.create_hnsw()
        print(f"Insert {num_elements} vectors to parallel index")
        start = time.time()
        index.add_vector_parallel(data, np.array(range(num_elements)), threads_num)
        dur = time.time() - start       
        print(f"Insert {num_elements} vectors to parallel index took {dur} s")
         
    create_parallel()
    def create_tiered():
        index = indices_ctx.create_tiered()
        
        print(f"Insert {num_elements} vectors to tiered index")
        print(f"flat buffer limit = {index.get_buffer_limit()}")
        start = time.time()
        for i, vector in enumerate(data):
            index.add_vector(vector, i)
        bf_dur = time.time() - start
        
        print(f''' insert to bf took {bf_dur}, current hnsw size is {index.hnsw_label_count()}")
                wait for index\n''')
        index.wait_for_index()
        dur = time.time() - start
        
        assert index.hnsw_label_count() == num_elements
        
        # Measure insertion to tiered index
        
        print(f"Insert {num_elements} vectors to tiered index took {dur} s")
        
        # Measure total memory of the tiered index 
        print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
    
    print(f"Start tiered hnsw creation")
    create_tiered()

def create_dbpedia_graph():
    indices_ctx = DBPediaIndexCtx()
    
    threads_num = TIEREDIndex.get_threads_num()
    print(f"thread num = {threads_num}")
    dbpeida_data = indices_ctx.data
    num_elements = indices_ctx.num_elements
    
    batches_num_per_ds = 10
    batch_size = int(num_elements / batches_num_per_ds)
    
       
    def create_tiered():
        index = indices_ctx.create_tiered()
        flat_buffer_limit = index.get_buffer_limit()
        print(f"flat buffer limit = {flat_buffer_limit}")
        assert flat_buffer_limit > batch_size
        
        #first insert dbpedia in batches
        for batch in range(batches_num_per_ds):
            print(f"Insert {batch_size} vectors from dbpedia to tiered index")
            first_label = batch * batch_size
            
            #insert in batches of batch size
            bf_time, start_wait = indices_ctx.insert_in_batch(index, dbpeida_data, data_first_idx= first_label, batch_size=batch_size, first_label = first_label)
            print(f''' insert to bf took {bf_time}, current hnsw size is {index.hnsw_label_count()}")
                    wait for index\n''')
            
            # measure time until wait for index for each batch
            index.wait_for_index()
            dur = time.time() - start_wait
            assert index.hnsw_label_count() == (batch + 1) * batch_size
            total_time = bf_time + dur
            print(f"Batch number {batch} : Insert {batch_size} vectors to tiered index took {total_time} s")
            print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
    
    
        #Next insert the random vactors
        for batch in range(batches_num_per_ds):
            print(f"Insert {batch_size} random vectors to tiered index")
            data_first_idx = batch * batch_size
            first_label = num_elements + data_first_idx
            
            #insert in batches of batch size
            bf_time, start_wait = indices_ctx.insert_in_batch(index, dbpeida_data, data_first_idx= data_first_idx, 
                                                              batch_size=batch_size, 
                                                              first_label = first_label)
            print(f''' insert to bf took {bf_time}, current hnsw size is {index.hnsw_label_count()}")
                    wait for index\n''')
            
            # measure time until wait for index for each batch
            index.wait_for_index()
            dur = time.time() - start_wait
            assert index.hnsw_label_count() == num_elements + (batch + 1 ) * batch_size
            total_time = bf_time + dur
            print(f"Batch number {batch} : Insert {batch_size} vectors to tiered index took {total_time} s")
            print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
    
    print(f"Start tiered hnsw creation")
    create_tiered()
    def create_hnsw():
        index = indices_ctx.create_hnsw()
        
        #first insert dbpedia in batches
        for batch in range(batches_num_per_ds):
            print(f"Insert {batch_size} vectors from dbpedia to sync hnsw index")
            first_label = batch * batch_size
            
            #insert in batches of batch size
            batch_time, _ = indices_ctx.insert_in_batch(index, dbpeida_data, data_first_idx= first_label, batch_size=batch_size, first_label = first_label)
            
            assert index.index_size() == (batch + 1) * batch_size
            print(f"Batch number {batch} : Insert {batch_size} vectors to tiered index took {batch_time} s")
            print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
        #first insert dbpedia in batches
        for batch in range(batches_num_per_ds):
            print(f"Insert {batch_size} vectors from dbpedia to sync hnsw index")
            data_first_idx = batch * batch_size
            first_label = num_elements + data_first_idx
            
            #insert in batches of batch size
            batch_time, _ = indices_ctx.insert_in_batch(index, dbpeida_data, data_first_idx= data_first_idx, batch_size=batch_size, first_label = first_label)
            
            assert index.index_size() == num_elements + (batch + 1) * batch_size
            print(f"Batch number {batch} : Insert {batch_size} vectors to tiered index took {batch_time} s")
            print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
   # print(f"dbpedia vectors = {dbpeida_data[0:4].shape[0]}")
  #  print(f"vectors = {vectors[0]}")
    print(f"Start hnsw creation")
    
    create_hnsw()


def insert_delete_reinsert():
    indices_ctx = DBPediaIndexCtx(data_size = 1000000,mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index
    threads_num = TIEREDIndex.get_threads_num()
    print(f"thread num = {threads_num}")
    data = indices_ctx.data
    num_elements = indices_ctx.num_elements
    
    # compute ground truth
    k = 10
    query_data = indices_ctx.generate_query_from_ds()
    
    bf_index = indices_ctx.init_and_populate_flat_index()
    bf_labels, _ = bf_index.knn_query(query_data, k)
    
    
    print(f"flat buffer limit = {index.get_buffer_limit()}")
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    bf_dur = time.time() - start
    
    print(f''' insert to bf took {bf_dur}, current hnsw size is {index.hnsw_label_count()}")
            wait for index\n''')
    index.wait_for_index()
    dur = time.time() - start
    
    query_start = time.time()
    tiered_labels, _ = index.knn_query(query_data, k)
    query_dur = time.time() - query_start
    
    print(f"query time = {round_ms(query_dur)} ms")
    
    curr_correct = len(np.intersect1d(tiered_labels[0], bf_labels[0]))   
    curr_recall = float(curr_correct)/k
    print(f"curr recall after firsdt insertion= {curr_recall}")
    
    # Delete half of the index.
    for i in range(0, num_elements, 2):
        index.delete_vector(i)
    assert index.hnsw_label_count() == (num_elements / 2)    
    index.wait_for_index()
    
    #reinsert the deleted vectors
    start = time.time()
    for i in range(0, num_elements, 2):
        vector = data[i]
        index.add_vector(vector, i)
    bf_dur = time.time() - start
    
    print(f''' insert to bf took {bf_dur}, current hnsw size is {index.hnsw_label_count()}")
            wait for index\n''')
    index.wait_for_index()
    dur = time.time() - start 
    assert index.hnsw_label_count() == (num_elements)    
    print(f''' reinsert to the hnsw took insert to bf took {dur}, current hnsw size is {index.hnsw_label_count()}")''')   
    print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
    
    

    
    print(f"total memory of bf index = {bf_index.index_memory()/pow(10,9)} GB")
    
    query_start = time.time()
    tiered_labels, _ = index.knn_query(query_data, k)
    query_dur = time.time() - query_start
    
    print(f"query time = {round_ms(query_dur)} ms")
    
    curr_correct = len(np.intersect1d(tiered_labels[0], bf_labels[0]))   
    curr_recall = float(curr_correct)/k
    print(f"curr recall = {curr_recall}")
    
    
def insert_and_update():
    indices_ctx = DBPediaIndexCtx(mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index
    threads_num = TIEREDIndex.get_threads_num()
    print(f"thread num = {threads_num}")
    data = indices_ctx.data
    num_elements = indices_ctx.num_elements
    
    print(f"flat buffer limit = {index.get_buffer_limit()}")
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    bf_dur = time.time() - start
    
    print(f''' insert {num_elements} vecs to bf took {bf_dur}, current hnsw size is {index.hnsw_label_count()}")
            wait for index\n''')
    index.wait_for_index()
    dur = time.time() - start
    # Measure insertion to tiered index
    
    print(f"Insert {num_elements} vectors to tiered index took {dur} s")
    
    # Measure total memory of the tiered index 
    print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
    
    assert index.get_curr_bf_size() == 0
     
    def search_insert(is_multi: bool, num_per_label = 1):
        
        #choose random vector from the data base and perform query on it
        query_data = indices_ctx.generate_query_from_ds()
       # query_data = indices_ctx.generate_queries(num_queries=1)
        k = 10
        # Calculate ground truth results
        bf_index = indices_ctx.init_and_populate_flat_index()
        bf_labels, _ = bf_index.knn_query(query_data, k)
        
        assert bf_index.index_size() == num_elements
        def query():
            query_start = time.time()
            tiered_labels, _ = index.knn_query(query_data, k)
            query_dur = time.time() - query_start
            
            print(f"query time = {round_ms(query_dur)} ms")
            
            curr_correct = len(np.intersect1d(tiered_labels[0], bf_labels[0]))   
            curr_recall = float(curr_correct)/k
            print(f"curr recall = {curr_recall}")
            
            return query_dur, curr_correct
        # config knn log
        index.start_knn_log()
        
        
        # query before any changes
        print("query before overriding")
        _, _ = query()
        assert index.get_curr_bf_size(mode = 'insert_and_knn') == 0
        
        # Start background insertion to the tiered index.
        print(f"start overriding")
        index_start, bf_dur, _ = indices_ctx.populate_index(index)
        print(f"bf size is:" )
        bf_size = index.hnsw_label_count()
        print(f"{bf_size}")
        print(f"current hnsw size is {index.hnsw_label_count()}")
        print(f"insert to bf took {bf_dur}")
        print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
        
        correct = 0
        searches_number = 0
        
        # run knn query every 1 s. 
        total_tiered_search_time = 0
        bf_curr_size = num_elements
        while bf_curr_size != 0:
            query_dur, curr_correct = query()
            # For each run get the current hnsw size and the query time.
            total_tiered_search_time += query_dur
            bf_curr_size = index.get_curr_bf_size(mode = 'insert_and_knn')
            
            print(f"bf size = {bf_curr_size}")
            correct += curr_correct
             
            time.sleep(5)
            searches_number += 1
        
        index.reset_log()
        
        # HNSW labels count updates before the job is done, so we need to wait for the queue to be empty.
        index.wait_for_index(1)
        index_dur = time.time() - index_start
        assert index.get_curr_bf_size() == 0
        assert index.hnsw_label_count() == num_elements

        print(f"indexing during search in tiered took {round_(index_dur)} s, all repair jobs are done")
        print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
        
        # Measure recall.
        recall = float(correct)/(k*searches_number)
        print("Average recall is:", round_(recall, 3))
        print("tiered query per seconds: ", round_(searches_number/total_tiered_search_time))  
        
        #execute swap jobs and execute query
        swap_start = time.time()
        
        index.execute_swap_jobs()
        
        swap_dur = time.time() - swap_start
        print(f"swap jobs took = {round_ms(swap_dur)} ms")
        
        assert index.hnsw_marked_deleted() == 0 
        print("query after swap execution")
        print(f"total memory of tiered index = {index.index_memory()/pow(10,9)} GB")
        
        
        query_dur, curr_correct = query()
        
    search_insert(is_multi=False)
    
    
def test_main():
    print("Test creation")
  #  create_dbpedia()
  #  create_dbpedia_graph()
    print(f"\nStart insert & search test")
   # search_insert(is_multi=False)
    insert_and_update()
    #or sanity
    #insert_delete_reinsert()
  

