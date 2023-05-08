# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import time
from common import *


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
                 ef_r = 20, 
                 metric = VecSimMetric_Cosine, 
                 data_type = VecSimType_FLOAT32, 
                 is_multi = False, 
                 num_per_label = 1,
                 swap_job_threshold = 0):
        self.num_vectors = data_size
        self.dim = dim
        self.M = M
        self.efConstruction = ef_c
        self.efRuntime = ef_r 
        self.metric = metric
        self.data_type = data_type
        self.is_multi = is_multi
        self.num_per_label = num_per_label
        
        # Generate data.
        self.num_labels = int(self.num_vectors/num_per_label)
        
        self.rng = np.random.default_rng(seed=47)
        
        data_shape = (self.num_labels, num_per_label, self.dim) if is_multi else (self.num_labels, self.dim)
        data = self.rng.random(data_shape) 
        self.data = np.float32(data) if self.data_type == VecSimType_FLOAT32 else data

        
        self.hnsw_params = create_hnsw_params(dim = self.dim, 
                                              num_elements = self.num_vectors, 
                                              metric = self.metric,
                                              data_type = self.data_type,
                                              ef_construction = ef_c,
                                              m = M,
                                              ef_runtime = ef_r,
                                              is_multi = self.is_multi)
        self.tiered_hnsw_params = create_tiered_hnsw_params(swap_job_threshold)
        
        self.tiered_index = Tiered_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params)
    
    def populate_index_multi(self, index):
        start = time.time()
        duration = 0
        for label, vectors in enumerate(self.data):
            for vector in vectors:
                start_add = time.time()
                index.add_vector(vector, label)
                duration += time.time() - start_add
        end = time.time()
        return (start, duration, end)
    
    def populate_index(self, index):
        if self.is_multi:
            return self.populate_index_multi(index)
        start = time.time()
        duration = 0
        for label, vector in enumerate(self.data):
            start_add = time.time()
            index.add_vector(vector, label)
            duration += time.time() - start_add
        end = time.time()
        return (start, duration, end)
                    
    def init_and_populate_flat_index(self):
        bfparams = BFParams()
        bfparams.initialCapacity = self.num_vectors
        bfparams.dim = self.dim
        bfparams.type = self.data_type
        bfparams.metric = self.metric
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
        queries = self.rng.random((num_queries, self.dim)) 
        return np.float32(queries) if self.data_type == VecSimType_FLOAT32 else queries
    
    def get_vectors_memory_size(self):
        data_type_size = 4 if self.data_type == VecSimType_FLOAT32 else 8
        return bytes_to_mega(self.num_vectors * self.dim * data_type_size)
           
def create_tiered_index(is_multi: bool, num_per_label = 1):
    indices_ctx = IndexCtx(data_size=50000, is_multi=is_multi, num_per_label=num_per_label)
    num_elements = indices_ctx.num_labels
    
    threads_num = TieredIndex.get_threads_num()
    
    # Initialize time measurements to apply assert at the end.
    tiered_index_time = 0
    hnsw_index_time = 0
    
    # Create the tiered index
    index = indices_ctx.tiered_index
    
    _, bf_dur, end_add_time = indices_ctx.populate_index(index)
    
    index.wait_for_index()
    tiered_index_time = bf_dur + time.time() - end_add_time
    
    assert index.hnsw_label_count() == num_elements
    
    # Measure insertion to tiered index.
    print(f"Insert {num_elements} insertion jobs took {round_ms(bf_dur)} ms")
    print(f"Done insertion jobs executions using {threads_num} threads took {round_ms(tiered_index_time)} ms")
    
    # Measure total memory of the tiered index.
    tiered_memory = bytes_to_mega(index.index_memory())
    
    print(f"total memory of tiered index = {tiered_memory} MB")
    
    hnsw_index = HNSWIndex(indices_ctx.hnsw_params)
    _, hnsw_index_time, _ = indices_ctx.populate_index(hnsw_index)

    print(f"Insert {num_elements} vectors to sync hnsw index took {round_(hnsw_index_time)} s")   
    hnsw_memory = bytes_to_mega(hnsw_index.index_memory())
    print(f"total memory of hnsw index = {hnsw_memory} MB")
    
    # The index memory should be at least as the total memory of the vectors.
    assert hnsw_memory > indices_ctx.get_vectors_memory_size()
    
    # Tiered index memory should be greater than HNSW index memory.
    assert tiered_memory > hnsw_memory
    execution_time_ratio = hnsw_index_time / tiered_index_time
    print(f"with {threads_num} threads, insertion runtime is {round_(execution_time_ratio)} times better \n")
    

def search_insert(is_multi: bool, num_per_label = 1):
    indices_ctx = IndexCtx(data_size=100000, is_multi=is_multi, num_per_label=num_per_label)
    index = indices_ctx.tiered_index

    num_labels = indices_ctx.num_labels
    
    query_data = indices_ctx.generate_queries(num_queries=1)
    
    # Add vectors to the flat index.
    bf_index = indices_ctx.init_and_populate_flat_index()
    
    # Start background insertion to the tiered index.
    index_start, _, _ = indices_ctx.populate_index(index)
    
    correct = 0
    k = 10
    searches_number = 0
    
    # run knn query every 1 s. 
    total_tiered_search_time = 0
    prev_bf_size = num_labels
    while index.hnsw_label_count() < num_labels:
        # For each run get the current hnsw size and the query time.
        bf_curr_size = index.get_curr_bf_size()
        query_start = time.time()
        tiered_labels, _ = index.knn_query(query_data, k)
        query_dur = time.time() - query_start
        total_tiered_search_time += query_dur
        
        print(f"query time = {round_ms(query_dur)} ms")
        
        # BF size should decrease.
        print(f"bf size = {bf_curr_size}")
        assert bf_curr_size < prev_bf_size
        
        # Run the query also in the bf index to get the ground truth results.
        bf_labels, _ = bf_index.knn_query(query_data, k)
        correct += len(np.intersect1d(tiered_labels[0], bf_labels[0]))    
        time.sleep(1)
        searches_number += 1
        prev_bf_size = bf_curr_size
    
    # HNSW labels count updates before the job is done, so we need to wait for the queue to be empty.
    index.wait_for_index(1)
    index_dur = time.time() - index_start
    print(f"indexing during search in tiered took {round_(index_dur)} s")
    
    # Measure recall.
    recall = float(correct)/(k*searches_number)
    print("Average recall is:", round_(recall, 3))
    print("tiered query per seconds: ", round_(searches_number/total_tiered_search_time)) 


def test_create_tiered():
    print("\nTest create tiered hnsw index")
    create_tiered_index(is_multi=False)      
      
def test_create_multi():
    print("Test create multi label tiered hnsw index")
    
def test_search_insert():
    print(f"\nStart insert & search test")
    search_insert(is_multi=False)
    
def test_search_insert_multi_index():
    print(f"\nStart insert & search test for multi index")
    
    search_insert(is_multi=True, num_per_label=5)
    create_tiered_index(is_multi=True, num_per_label=5)
    
# In this test we insert the vectors one by one to the tiered index (call wait_for_index after each add vector)
# We expect to get the same index as if we were inserting the vector to the sync hnsw index.
# To check that, we perform a knn query with k = vectors number and compare the results' labels
# to pass the test all the labels and distances should be the same.
def test_sanity():
    
    indices_ctx = IndexCtx()
    index = indices_ctx.tiered_index    
    k = indices_ctx.num_labels
    
    print(f"\nadd {indices_ctx.num_labels} vectors to the tiered index one by one")
    # Add vectors to the tiered index one by one.
    for i, vector in enumerate(indices_ctx.data):
        index.add_vector(vector, i)
        index.wait_for_index(1)
    
    assert index.hnsw_label_count() == indices_ctx.num_labels
    
    # Create hnsw index.
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()
    
    query_data = indices_ctx.generate_queries(num_queries=1)
    
    # Search knn in tiered.
    tiered_labels, tiered_dist = index.knn_query(query_data, k)
    # Search knn in hnsw.
    hnsw_labels, hnsw_dist = hnsw_index.knn_query(query_data, k)
    
    # Compare.
    has_diff = False
    for i, hnsw_res_label in enumerate(hnsw_labels[0]):
        if hnsw_res_label != tiered_labels[0][i]:
            has_diff = True
            print(f"hnsw label = {hnsw_res_label}, tiered label = {tiered_labels[0][i]}")
            print(f"hnsw dist = {hnsw_dist[0][i]}, tiered dist = {tiered_dist[0][i]}")

    assert has_diff == False
    print(f"hnsw graph is identical to the tiered index graph")
    
def test_recall_after_deletion():
    
    indices_ctx = IndexCtx(ef_r=20)
    index = indices_ctx.tiered_index
    data = indices_ctx.data
    num_elements = indices_ctx.num_labels
    
    # Create hnsw index.
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()
    
    print(f"\nadd {indices_ctx.num_labels} vectors to the tiered index one by one")
    
    # Populate tiered index.
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    index.wait_for_index()
    
    print(f"Delete half of the index")
    # Delete half of the index.
    for i in range(0, num_elements, 2):
        index.delete_vector(i)
        hnsw_index.delete_vector(i)
        
    # Wait for all repair jobs to be done.
    index.wait_for_index(5)
    
    assert index.hnsw_label_count() == (num_elements / 2)
    assert hnsw_index.index_size() == (num_elements / 2)
    
    # Create a list of tuples of the vectors that left.
    vectors = [vectors[i] for i in range(1, num_elements, 2)]
    
    # Perform queries.
    num_queries = 10
    queries = indices_ctx.generate_queries(num_queries=10)
    
    k = 10
    correct_tiered = 0
    correct_hnsw = 0
    
    # Calculate correct vectors for each index.
    # We don't expect hnsw and tiered hnsw results to be identical due to the parallel insertion.
    def calculate_correct(index_labels, keys):
        correct = 0;
        for label in index_labels[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct += 1
                    break 
        return correct
    
    for target_vector in queries:
        tiered_labels, _ = index.knn_query(target_vector, k)
        hnsw_labels, _ = hnsw_index.knn_query(target_vector, k)
        
        # Sort distances of every vector from the target vector and get actual k nearest vectors.
        dists = [(spatial.distance.cosine(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        keys = [key for _, key in dists[:k]]
        correct_tiered += calculate_correct(tiered_labels, keys)
        correct_hnsw += calculate_correct(hnsw_labels, keys)

    # Measure recall.
    recall_tiered = float(correct_tiered) / (k * num_queries)
    recall_hnsw = float(correct_hnsw) / (k * num_queries)
    print("hsnw tiered recall is: \n", recall_tiered)
    print("hsnw recall is: \n", recall_hnsw)
    assert (recall_tiered >= 0.9)


def test_batch_iterator():
    num_elements = 100000
    dim = 100
    M = 26
    efConstruction = 180
    efRuntime = 180
    metric = VecSimMetric_L2
    indices_ctx = IndexCtx(data_size=num_elements, 
                           dim=dim, 
                           M=M, 
                           ef_c=efConstruction, 
                           ef_r=efRuntime, 
                           metric=metric)

    index = indices_ctx.tiered_index
    data = indices_ctx.data
    
    print(f"\n Test batch iterator in tiered index")
    
    vectors = []
    # Add 100k random vectors to the index.
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))
    
    # Create a random query vector and create a batch iterator.
    query_data = indices_ctx.generate_queries(num_queries=1)
    batch_iterator = index.create_batch_iterator(query_data)
    batch_size = 10
    labels_first_batch, distances_first_batch = batch_iterator.get_next_results(batch_size, BY_ID)
    
    for i, _ in enumerate(labels_first_batch[0][:-1]):
        # Assert sorting by id.
        assert (labels_first_batch[0][i] < labels_first_batch[0][i + 1])

    labels_second_batch, distances_second_batch = batch_iterator.get_next_results(batch_size, BY_SCORE)
    should_have_return_in_first_batch = []
    for i, dist in enumerate(distances_second_batch[0][:-1]):
        # Assert sorting by score.
        assert (distances_second_batch[0][i] < distances_second_batch[0][i + 1])
        # Assert that every distance in the second batch is higher than any distance of the first batch.
        if len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) != 0:
            should_have_return_in_first_batch.append(dist)
    assert (len(should_have_return_in_first_batch) <= 2)

    # Verify that runtime args are sent properly to the batch iterator.
    query_params = VecSimQueryParams()
    query_params.hnswRuntimeParams.efRuntime = 5
    
    batch_iterator_new = index.create_batch_iterator(query_data, query_params)
    
    labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(batch_size, BY_ID)
    # Verify that accuracy is worse with the new lower ef_runtime.
    assert (sum(labels_first_batch[0]) <= sum(labels_first_batch_new[0]))

    query_params.hnswRuntimeParams.efRuntime = efRuntime  # Restore previous ef_runtime.
    batch_iterator_new = index.create_batch_iterator(query_data, query_params)
    labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(batch_size, BY_ID)
    # Verify that results are now the same.
    assert_allclose(labels_first_batch_new[0], labels_first_batch[0])

    # Reset.
    batch_iterator.reset()

    # Run in batches of 100 until we reach 1000 results and measure recall.
    batch_size = 100
    total_res = 1000
    total_recall = 0
    num_queries = 10
    query_data = indices_ctx.generate_queries(num_queries=num_queries)
    for target_vector in query_data:
        correct = 0
        batch_iterator = index.create_batch_iterator(target_vector)
        iterations = 0
        # Sort distances of every vector from the target vector and get the actual order.
        dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        accumulated_labels = []
        while batch_iterator.has_next():
            iterations += 1
            labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
            accumulated_labels.extend(labels[0])
            returned_results_num = len(accumulated_labels)
            if returned_results_num == total_res:
                keys = [key for _, key in dists[:returned_results_num]]
                correct += len(set(accumulated_labels).intersection(set(keys)))
                break
        assert iterations == np.ceil(total_res / batch_size)
        recall = float(correct) / total_res
        assert recall >= 0.89
        total_recall += recall
    print(f'\nAvg recall for {total_res} results in index of size {num_elements} with dim={dim} is: ',
          round_(total_recall / num_queries))

    # Run again a single query in batches until it is depleted.
    batch_iterator = index.create_batch_iterator(query_data[0])
    iterations = 0
    accumulated_labels = set()

    while batch_iterator.has_next():
        iterations += 1
        labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
        # Verify that we got new scores in each iteration.
        assert len(accumulated_labels.intersection(set(labels[0]))) == 0
        accumulated_labels = accumulated_labels.union(set(labels[0]))
    assert len(accumulated_labels) >= 0.95 * num_elements
    print("Overall results returned:", len(accumulated_labels), "in", iterations, "iterations")

