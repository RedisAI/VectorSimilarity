# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import time
from common import *
from enum import Enum

class CreationMode(Enum):
    ONLY_PARAMS = 1
    CREATE_TIERED_INDEX = 2 

# swap_job_threshold = 0 means use the default swap_job_threshold defined in hnsw_tiered.h
def create_tiered_hnsw_params(swap_job_threshold = 0):
    tiered_hnsw_params = TieredHNSWParams()
    tiered_hnsw_params.swapJobThreshold  = swap_job_threshold
    return tiered_hnsw_params   

class IndexCtx:
    def __init__(self, data_size = 10000, 
                 dim = 16,                 
                 M = 16, 
                 ef_c = 512, 
                 ef_r = 10, 
                 metric = VecSimMetric_Cosine, 
                 data_type = VecSimType_FLOAT32, 
                 is_multi = False, 
                 num_per_label = 1,
                 swap_job_threshold = 0, 
                 mode = CreationMode.ONLY_PARAMS):
        self.num_vectors = data_size
        self.dim = dim
        self.M = M
        self.efConstruction = ef_c
        self.efRuntime = ef_r 
        self.metric = metric
        self.data_type = data_type
        self.is_multi = is_multi
        self.num_per_label = num_per_label
        
        # generate data
        self.num_labels = int(self.num_vectors/num_per_label)
        data_shape = (self.num_labels, num_per_label, self.dim)
        
        data = np.random.random(data_shape) 
        self.data = np.float32(data) if self.data_type == VecSimType_FLOAT32 else data

        
        self.hnsw_params = create_hnsw_params(dim=self.dim, 
                                              num_elements=self.num_vectors, 
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
        
    def populate_index(self, index):
        start = time.time()
        duration = 0
        for label, vectors in enumerate(self.data):
            for vector in vectors:
                start_add = time.time()
                index.add_vector(vector, label)
                duration += time.time() - start_add
        end = time.time()
        return (start, duration, end)
                    
    def init_and_populate_flat_index(self):
        bfparams = BFParams()
        bfparams.initialCapacity = self.num_vectors
        bfparams.dim =self.dim
        bfparams.type =self.data_type
        bfparams.metric =self.metric
        bfparams.multi = self.is_multi
        self.flat_index = BFIndex(bfparams)
        
        self.populate_index(self.flat_index)
        
        return self.flat_index
    
    def create_hnsw_index(self):
        return HNSWIndex(self.hnsw_params)
    
    def init_and_populate_hnsw_index(self):
        hnsw_index = HNSWIndex(self.hnsw_params)
        self.hnsw_index = hnsw_index
        
        self.populate_index(hnsw_index)
        return hnsw_index
    
    def generate_queries(self, num_queries):
        queries = np.random.random((num_queries, self.dim)) 
        return np.float32(queries) if self.data_type == VecSimType_FLOAT32 else queries
    
    def get_num_labels(self):
        return self.num_vectors / self.num_per_label
        
    
def test_create():
    indices_ctx = IndexCtx()
    num_elements = indices_ctx.num_vectors
    
    threads_num = TIEREDIndex.get_threads_num()
    
    # Initialize time measurements to apply assert at the end.
    tiered_index_time = 0
    hnsw_index_time = 0
    
    def create_tiered():
        nonlocal tiered_index_time
        index = indices_ctx.create_tiered()
        
        _, bf_dur, end_add_time = indices_ctx.populate_index(index)
        
        index.wait_for_index()
        tiered_index_time = bf_dur + time.time() - end_add_time
        
        assert index.hnsw_label_count() == num_elements
        
        # Measure insertion to tiered index
        
        print(f"\nAdding {num_elements} insertion jobs took {round_(bf_dur)}")
        print(f"Insert vectors to the tiered index using {threads_num} threads took {round_(tiered_index_time)} s")
        
        # Measure total memory of the tiered index 
        print(f"total memory of tiered index = {bytes_to_giga(index.index_memory())} GB")
    
    create_tiered()
    
    def create_hnsw():
        nonlocal hnsw_index_time

        hnsw_index = HNSWIndex(indices_ctx.hnsw_params)
        _, hnsw_index_time, _ = indices_ctx.populate_index(hnsw_index)

        print(f"Insert {num_elements} vectors to hnsw index took {round_(hnsw_index_time)} s")   
        print(f"total memory of hnsw index = {bytes_to_giga(hnsw_index.index_memory())} GB\n")
    
    create_hnsw()
    
    parallel_execution_portion = 0.85  # we expect that at least 85% of the insert time will be executed in parallel
    
    execution_time_ratio = hnsw_index_time / tiered_index_time
    expected_insertion_speedup = round_(expected_speedup(parallel_execution_portion, threads_num), 3)
    print(f"Expected insertion speedup is {expected_insertion_speedup}")
    print(f"with {threads_num} threads, improvement ratio in insertion runtime is {round_(execution_time_ratio)} \n")
    
    assert execution_time_ratio > expected_insertion_speedup

def search_insert(is_multi: bool, num_per_label = 1):
    indices_ctx = IndexCtx(mode=CreationMode.CREATE_TIERED_INDEX, is_multi=is_multi, num_per_label=num_per_label)
    index = indices_ctx.tiered_index

    num_labels = indices_ctx.get_num_labels()
    
    query_data = indices_ctx.generate_queries(num_queries=1)
    
    # Add vectors to the flat index.
    bf_index = indices_ctx.init_and_populate_flat_index()
    
    # Start background insertion to the tiered index
    index_start, _, _ = indices_ctx.populate_index(index)
    
    correct = 0
    k = 10
    searches_number = 0
    
    # config the index log to knn mode to get access to the bf index size during the query.
    index.start_knn_log()
    
    # run knn query every 1 s 
    total_tiered_search_time = 0
    prev_query_duration = 0
    prev_bf_size = 0
    while index.hnsw_label_count() < num_labels:
        query_start = time.time()
        tiered_labels, _ = index.knn_query(query_data, k)
        query_dur = time.time() - query_start
        total_tiered_search_time += query_dur
        # for each run get the current hnsw size and the query time
        bf_curr_size = index.get_curr_bf_size(mode = 'insert_and_knn')
        
        # expect query time to improve
        print(f"query time = {round_(query_dur)}")
        assert query_dur > prev_query_duration
        
        # while bf size should decrease
        print(f"bf size = {bf_curr_size}")
        assert bf_curr_size > prev_bf_size
        
        # run the query also in the bf index to get the ground truth results
        bf_labels, _ = bf_index.knn_query(query_data, k)
        correct += len(np.intersect1d(tiered_labels[0], bf_labels[0]))    
        time.sleep(1)
        searches_number += 1
        prev_query_duration = query_dur
        prev_bf_size = bf_curr_size
    
    # hnsw labels count updates before the job is done, so we need to wait for the queue to be empty
    index.wait_for_index(1)
    index_dur = time.time() - index_start
    print(f"indexing during search in tiered took {round_(index_dur)} s")
    
    # Measure recall
    recall = float(correct)/(k*searches_number)
    print("Average recall is:", round_(recall, 3))
    print("tiered query per seconds: ", round_(searches_number/total_tiered_search_time)) 

def test_search_insert():
    print(f"\nStart insert & search test")
    search_insert(is_multi = False)
    
def test_search_insert_multi_index():
    print(f"\nStart insert & search test for multi index")
    
    search_insert(is_multi = True, num_per_label=5)
    
# In this test we insert the vectors one by one to the tiered index (call wait_for_index after each add vector)
# We expect to get the same index as if we were inserting the vector to the sync hnsw index.
# To check that, we perform a knn query with k = vectors number and compare the results' labels
# to pass the test all the labels and distances should be the same.
def test_sanity():
    
    indices_ctx = IndexCtx(mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index    
    k = IndexCtx.num_vectors
    
    #add vectors to the tiered index one by one
    for i, vector in enumerate(indices_ctx.data):
        index.add_vector(vector, i)
        index.wait_for_index()
    
    print(f"done indexing, label count = {index.hnsw_label_count()}")
    assert index.hnsw_label_count() == indices_ctx.num_labels
    
    # create hnsw index
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()
    
    query_data = indices_ctx.generate_queries(num_queries=1)

    #search knn in tiered
    tiered_labels, tiered_dist = index.knn_query(query_data, k)
    #search knn in hnsw
    hnsw_labels, hnsw_dist = hnsw_index.knn_query(query_data, k)
    
    #compare
    has_diff = False
    for i, hnsw_res_label in enumerate(hnsw_labels[0]):
        if hnsw_res_label != tiered_labels[0][i]:
            has_diff = True
            print(f"hnsw label = {hnsw_res_label}, tiered label = {tiered_labels[0][i]}")
            print(f"hnsw dist = {hnsw_dist[0][i]}, tiered dist = {tiered_dist[0][i]}")

    assert has_diff == False
    

def recall_after_deletion():
    
    indices_ctx = IndexCtx(mode=CreationMode.CREATE_TIERED_INDEX, ef_r=20)
    index = indices_ctx.tiered_index
    data = indices_ctx.data
    num_elements = indices_ctx.num_labels
    
    #create hnsw index
    hnsw_index = create_hnsw_index()
    
    # Populate tiered index 
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    print(f"current flat buffer size is {index.get_curr_bf_size()}, wait for index\n")
    index.wait_for_index()
    
    
    #delete half of the index
    for i in range(0, num_elements, 2):
        index.delete_vector(i)
        hnsw_index.delete_vector(i)
        
    # wait for all repair jobs to be done
    index.wait_for_index(5)
    
    assert index.hnsw_label_count() == (num_elements / 2)
    assert hnsw_index.index_size() == (num_elements / 2)
    
    #create a list of tuples of the vectors that left
    vectors = [vectors[i] for i in range(1, num_elements, 2)]
    
    # perform queries
    num_queries = 10
    queries = indices_ctx.generate_queries(num_queries=10)
    
    k = 10
    correct_tiered = 0
    correct_hnsw = 0
    
    # calculate correct vectors for each index  
    # We don't expect hnsw and tiered hnsw results to be identical due to the parallel insertion.
    def calculate_correct(index_labels, keys):
        correct = 0;
        for label in index_labels[0]:
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
    assert (recall_tiered >= recall_hnsw)


#TODO!!!!!!
def create_multi_index():
    indices_ctx = IndexCtx(mode=CreationMode.CREATE_TIERED_INDEX)
    index = indices_ctx.tiered_index
    
    num_elements = indices_ctx.num_vectors
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
