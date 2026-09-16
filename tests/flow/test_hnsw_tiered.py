# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).
import time
import pytest
from common import *


# swap_job_threshold = 0 means use the default swap_job_threshold defined in hnsw_tiered.h
def create_tiered_hnsw_params(swap_job_threshold = 0):
    tiered_hnsw_params = TieredHNSWParams()
    tiered_hnsw_params.swapJobThreshold = swap_job_threshold
    return tiered_hnsw_params

class IndexCtx:
    array_conversion_func = {
        VecSimType_FLOAT32: np.float32,
        VecSimType_BFLOAT16: vec_to_bfloat16,
        VecSimType_FLOAT16: vec_to_float16,
    }

    type_to_dtype = {
        VecSimType_FLOAT32: np.float32,
        VecSimType_FLOAT64: np.float64,
        VecSimType_BFLOAT16: bfloat16,
        VecSimType_FLOAT16: np.float16,
        VecSimType_INT8: np.int8,
        VecSimType_UINT8: np.uint8,
    }

    def __init__(self, data_size=10000,
                 dim=16,
                 M=16,
                 ef_c=512,
                 ef_r=20,
                 metric=VecSimMetric_Cosine,
                 data_type=VecSimType_FLOAT32,
                 is_multi=False,
                 num_per_label=1,
                 swap_job_threshold=0,
                 flat_buffer_size=1024,
                 create_data_func = None):
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
        self.create_data_func = self.rng.random if create_data_func is None else create_data_func

        data_shape = (self.num_labels, num_per_label, self.dim) if is_multi else (self.num_labels, self.dim)


        self.data = self.create_data_func(data_shape)
        if self.data_type in self.array_conversion_func.keys():
            self.data = self.array_conversion_func[self.data_type](self.data)
        # Note: data type logging moved to test functions that use this class
        assert self.data.dtype == self.type_to_dtype[self.data_type]

        self.hnsw_params = create_hnsw_params(dim = self.dim,
                                              num_elements = self.num_vectors,
                                              metric = self.metric,
                                              data_type = self.data_type,
                                              ef_construction = ef_c,
                                              m = M,
                                              ef_runtime = ef_r,
                                              is_multi = self.is_multi)
        self.tiered_hnsw_params = create_tiered_hnsw_params(swap_job_threshold)

        self.tiered_index = Tiered_HNSWIndex(self.hnsw_params, self.tiered_hnsw_params, flat_buffer_size)

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
        queries = self.create_data_func((num_queries, self.dim))
        if self.data_type in self.array_conversion_func.keys():
            queries = self.array_conversion_func[self.data_type](queries)
        return queries

    def get_vectors_memory_size(self):
        memory_size = {
            VecSimType_FLOAT32: 4,
            VecSimType_FLOAT64: 8,
            VecSimType_BFLOAT16: 2,
            VecSimType_FLOAT16: 2,
            VecSimType_INT8: 1,
            VecSimType_UINT8: 1,
        }
        return bytes_to_mega(self.num_vectors * self.dim * memory_size[self.data_type])

def create_tiered_index(test_logger, is_multi: bool, num_per_label=1, data_type=VecSimType_FLOAT32, create_data_func=None):
    indices_ctx = IndexCtx(data_size=50000, is_multi=is_multi, num_per_label=num_per_label, data_type=data_type, create_data_func=create_data_func)
    test_logger.info(f"data type = {indices_ctx.data.dtype}")
    num_elements = indices_ctx.num_labels

    index = indices_ctx.tiered_index
    threads_num = index.get_threads_num()

    _, bf_dur, end_add_time = indices_ctx.populate_index(index)

    index.wait_for_index()
    tiered_index_time = bf_dur + time.time() - end_add_time

    assert index.hnsw_label_count() == num_elements

    # Measure insertion to tiered index.
    test_logger.info(f"Insert {num_elements} vectors into the flat buffer took {round_ms(bf_dur)} ms")
    test_logger.info(f"Total time for inserting vectors to the tiered index and indexing them into HNSW using {threads_num}"
                     f" threads took {round_ms(tiered_index_time)} ms")

    # Measure total memory of the tiered index.
    tiered_memory = bytes_to_mega(index.index_memory())

    test_logger.info(f"total memory of tiered index = {tiered_memory} MB")

    hnsw_index = HNSWIndex(indices_ctx.hnsw_params)
    _, hnsw_index_time, _ = indices_ctx.populate_index(hnsw_index)

    test_logger.info(f"Insert {num_elements} vectors directly to HNSW index (one by one) took {round_(hnsw_index_time)} s")
    hnsw_memory = bytes_to_mega(hnsw_index.index_memory())
    test_logger.info(f"total memory of hnsw index = {hnsw_memory} MB")

    # The index memory should be at least as the total memory of the vectors.
    assert hnsw_memory > indices_ctx.get_vectors_memory_size()

    # Tiered index memory should be greater than HNSW index memory.
    assert tiered_memory > hnsw_memory
    execution_time_ratio = hnsw_index_time / tiered_index_time
    test_logger.info(f"with {threads_num} threads, insertion runtime is {round_(execution_time_ratio)} times better")


def search_insert(test_logger, is_multi: bool, num_per_label=1, data_type=VecSimType_FLOAT32, create_data_func=None):
    data_size = 100000
    indices_ctx = IndexCtx(data_size=data_size, is_multi=is_multi, num_per_label=num_per_label,
                           flat_buffer_size=data_size, M=64, data_type=data_type, create_data_func=create_data_func)
    index = indices_ctx.tiered_index

    num_labels = indices_ctx.num_labels

    test_logger.info(f'''Insert total of {num_labels} {indices_ctx.data.dtype} vectors of dim = {indices_ctx.dim},
          {num_per_label} vectors in each label. Total labels = {num_labels}''')

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
    cur_hnsw_label_count = index.hnsw_label_count()
    if cur_hnsw_label_count == num_labels:
        test_logger.info("All vectors were already indexed into HNSW - cannot test search while indexing")
        assert False

    test_logger.info("Start running queries while indexing is done in the background")
    test_logger.info(f"HNSW labels number = {cur_hnsw_label_count}")
    while cur_hnsw_label_count < num_labels:
        # For each run get the current hnsw size and the query time.
        bf_curr_size = index.get_curr_bf_size()
        query_start = time.time()
        tiered_labels, _ = index.knn_query(query_data, k)
        query_dur = time.time() - query_start
        total_tiered_search_time += query_dur

        test_logger.info(f"query time = {round_ms(query_dur)} ms")

        # BF size should decrease.
        test_logger.info(f"bf size = {bf_curr_size}")
        assert bf_curr_size < prev_bf_size

        # Run the query also in the bf index to get the ground truth results.
        bf_labels, _ = bf_index.knn_query(query_data, k)
        correct += len(np.intersect1d(tiered_labels[0], bf_labels[0]))
        time.sleep(1)
        searches_number += 1
        prev_bf_size = bf_curr_size
        cur_hnsw_label_count = index.hnsw_label_count()

    # HNSW labels count updates before the job is done, so we need to wait for the queue to be empty.
    index.wait_for_index(1)
    index_dur = time.time() - index_start
    test_logger.info(f"Indexing during searching in the tiered index took {round_(index_dur)} s")

    # Measure recall.
    recall = float(correct)/(k*searches_number)
    test_logger.info(f"Average recall is: {round_(recall, 3)}")
    test_logger.info(f"tiered query per seconds: {round_(searches_number/total_tiered_search_time)}")


def test_create_tiered(test_logger):
    test_logger.info("Test create tiered hnsw index")
    create_tiered_index(test_logger, is_multi=False)

def test_create_multi(test_logger):
    test_logger.info("Test create multi label tiered hnsw index")
    create_tiered_index(test_logger, is_multi=True, num_per_label=5)

def test_create_bf16(test_logger):
    test_logger.info("Test create BFLOAT16 tiered hnsw index")
    create_tiered_index(test_logger, is_multi=False, data_type=VecSimType_BFLOAT16)

def test_create_fp16(test_logger):
    test_logger.info("Test create FLOAT16 tiered hnsw index")
    create_tiered_index(test_logger, is_multi=False, data_type=VecSimType_FLOAT16)

def test_create_int8(test_logger):
    test_logger.info("Test create INT8 tiered hnsw index")
    create_tiered_index(test_logger, is_multi=False, data_type=VecSimType_INT8, create_data_func=create_int8_vectors)

def test_create_uint8(test_logger):
    test_logger.info("Test create UINT8 tiered hnsw index")
    create_tiered_index(test_logger, is_multi=False, data_type=VecSimType_UINT8, create_data_func=create_uint8_vectors)

def test_search_insert(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False)

def test_search_insert_bf16(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_BFLOAT16)

def test_search_insert_fp16(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_FLOAT16)

def test_search_insert_int8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_INT8, create_data_func=create_int8_vectors)

def test_search_insert_uint8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_UINT8, create_data_func=create_uint8_vectors)

def test_search_insert_multi_index(test_logger):
    test_logger.info("Start insert & search test for multi index")

    search_insert(test_logger, is_multi=True, num_per_label=5)

# In this test we insert the vectors one by one to the tiered index (call wait_for_index after each add vector)
# We expect to get the same index as if we were inserting the vector to the sync hnsw index.
# To check that, we perform a knn query with k = vectors number and compare the results' labels
# to pass the test all the labels and distances should be the same.
def test_sanity(test_logger):

    indices_ctx = IndexCtx()
    index = indices_ctx.tiered_index
    k = indices_ctx.num_labels

    test_logger.info(f"add {indices_ctx.num_labels} vectors to the tiered index one by one")
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
            test_logger.info(f"hnsw label = {hnsw_res_label}, tiered label = {tiered_labels[0][i]}")
            test_logger.info(f"hnsw dist = {hnsw_dist[0][i]}, tiered dist = {tiered_dist[0][i]}")

    assert not has_diff
    test_logger.info(f"hnsw graph is identical to the tiered index graph")


def test_recall_after_deletion(test_logger):

    indices_ctx = IndexCtx(ef_r=30)
    index = indices_ctx.tiered_index
    data = indices_ctx.data
    num_elements = indices_ctx.num_labels

    # Create hnsw index.
    hnsw_index = indices_ctx.init_and_populate_hnsw_index()

    test_logger.info(f"add {indices_ctx.num_labels} vectors to the tiered index one by one")

    # Populate tiered index.
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    index.wait_for_index()

    test_logger.info(f"Deleting half of the index")
    # Delete half of the index.
    for i in range(0, num_elements, 2):
        index.delete_vector(i)
        hnsw_index.delete_vector(i)

    # Wait for all repair jobs to be done.
    index.wait_for_index(5)
    test_logger.info(f"Done deleting half of the index")
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
        correct = 0
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
    test_logger.info(f"HNSW tiered recall is: {recall_tiered}")
    test_logger.info(f"HNSW recall is: {recall_hnsw}")
    assert (recall_tiered >= 0.9)


def test_batch_iterator(test_logger):
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
                           metric=metric,
                           flat_buffer_size=num_elements)

    index = indices_ctx.tiered_index
    data = indices_ctx.data

    test_logger.info(f"Test batch iterator in tiered index")

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
    test_logger.info(f'Avg recall for {total_res} results in index of size {num_elements} with dim={dim} is: {round_(total_recall / num_queries)}')

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
    test_logger.info(f"Overall results returned: {len(accumulated_labels)} in {iterations} iterations")


def test_range_query(test_logger):
    num_elements = 100000
    dim = 100
    efConstruction = 200
    efRuntime = 10
    metric = VecSimMetric_L2

    indices_ctx = IndexCtx(data_size=num_elements,
                        dim=dim,
                        ef_c=efConstruction,
                        ef_r=efRuntime,
                        metric=metric)

    index = indices_ctx.tiered_index
    data = indices_ctx.data

    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = indices_ctx.generate_queries(num_queries=1)

    radius = 13.0
    recalls = {}

    for epsilon_rt in [0.001, 0.01, 0.1]:
        query_params = VecSimQueryParams()
        query_params.hnswRuntimeParams.epsilon = epsilon_rt
        start = time.time()
        tiered_labels, tiered_distances = index.range_query(query_data, radius=radius, query_param=query_params)
        end = time.time()
        res_num = len(tiered_labels[0])

        dists = sorted([(key, spatial.distance.sqeuclidean(query_data.flat, vec)) for key, vec in vectors])
        actual_results = [(key, dist) for key, dist in dists if dist <= radius]

        test_logger.info(
            f'lookup time for {num_elements} vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
            f' got {res_num} results, which are {res_num / len(actual_results)} of the entire results in the range.')

        # Compare the number of vectors that are actually within the range to the returned results.
        assert np.all(np.isin(tiered_labels, np.array([label for label, _ in actual_results])))

        assert max(tiered_distances[0]) <= radius
        recalls[epsilon_rt] = res_num / len(actual_results)

    # Expect zero results for radius==0
    tiered_labels, tiered_distances = index.range_query(query_data, radius=0)
    assert len(tiered_labels[0]) == 0


def test_multi_range_query(test_logger):
    num_labels = 20000
    per_label = 5
    num_elements = num_labels * per_label

    dim = 100
    efConstruction = 200
    efRuntime = 10
    metric = VecSimMetric_L2

    indices_ctx = IndexCtx(data_size=num_elements,
                        dim=dim,
                        ef_c=efConstruction,
                        ef_r=efRuntime,
                        metric=metric,
                        is_multi=True,
                        num_per_label=per_label)

    index = indices_ctx.tiered_index
    data = indices_ctx.data

    vectors = []
    for label, vecs in enumerate(data):
        for vector in vecs:
            index.add_vector(vector, label)
            vectors.append((label, vector))

    query_data = indices_ctx.generate_queries(num_queries=1)

    radius = 13.0
    recalls = {}
    # calculate distances of the labels in the index
    dists = {}
    for key, vec in vectors:
        dists[key] = min(spatial.distance.sqeuclidean(query_data.flat, vec), dists.get(key, np.inf))

    dists = list(dists.items())
    dists = sorted(dists, key=lambda pair: pair[1])
    keys = [key for key, dist in dists if dist <= radius]

    for epsilon_rt in [0.001, 0.01, 0.1]:
        query_params = VecSimQueryParams()
        query_params.hnswRuntimeParams.epsilon = epsilon_rt
        start = time.time()
        tiered_labels, tiered_distances = index.range_query(query_data, radius=radius, query_param=query_params)
        end = time.time()
        res_num = len(tiered_labels[0])

        test_logger.info(
            f'lookup time for ({num_labels} X {per_label}) vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
            f' got {res_num} results, which are {res_num / len(keys)} of the entire results in the range.')

        # Compare the number of vectors that are actually within the range to the returned results.
        assert np.all(np.isin(tiered_labels, np.array(keys)))

        # Asserts that all the results are unique
        assert len(tiered_labels[0]) == len(np.unique(tiered_labels[0]))

        assert max(tiered_distances[0]) <= radius
        recalls[epsilon_rt] = res_num / len(keys)

    # Expect higher recalls for higher epsilon values.
    assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

    # Expect zero results for radius==0
    tiered_labels, tiered_distances = index.range_query(query_data, radius=0)
    assert len(tiered_labels[0]) == 0


def test_relabel_vector(test_logger):
    dim = 16
    num_elements = 1000
    hnsw_params = create_hnsw_params(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32)
    # A flat buffer large enough to hold everything, so the relabel below has a real chance of
    # landing while the vector is still buffered with a pending ingest job.
    index = Tiered_HNSWIndex(hnsw_params, create_tiered_hnsw_params(), num_elements)

    data = np.float32(np.random.random((num_elements, dim)))
    for label, vector in enumerate(data):
        index.add_vector(vector, label)

    # Relabel one early and one late label. The workers ingest in insertion order, so by now the
    # early one is most likely already in HNSW while the late one is most likely still buffered
    # with a pending ingest job - between them the two tiers both get covered. The buffered case is
    # the delicate one: a job left holding the old label would either ingest the vector under it or
    # throw out of the worker thread, and neither would survive the assertions below.
    buffered = index.get_curr_bf_size()
    test_logger.info(f"relabeling with {buffered} of {num_elements} vectors still buffered")
    moved = {7: num_elements + 500, num_elements - 1: num_elements + 501}
    for old_label, new_label in moved.items():
        assert index.relabel_vector(old_label, new_label) == VecSimRelabel_OK

    index.wait_for_index()

    # Once ingestion has drained, the vector sits in HNSW under the new label and under no other.
    assert index.index_size() == num_elements
    assert index.hnsw_label_count() == num_elements
    for old_label, new_label in moved.items():
        assert_allclose(index.get_vector(new_label)[0], data[old_label], rtol=1e-6)
        assert index.get_vector(old_label).shape == (0, dim)

        labels, distances = index.knn_query(data[old_label], 1)
        assert labels[0][0] == new_label
        assert distances[0][0] < 1e-6

    # Each rejection is reported distinctly, and none of them modifies the index.
    assert index.relabel_vector(num_elements + 1, 0) == VecSimRelabel_OldLabelMissing
    assert index.relabel_vector(0, 1) == VecSimRelabel_NewLabelTaken
    assert index.relabel_vector(0, 0) == VecSimRelabel_SameLabel
    assert index.index_size() == num_elements
    test_logger.info("tiered relabel_vector moved the label across both tiers")


def test_relabel_vector_multi(test_logger):
    dim = 16
    num_labels = 200
    per_label = 5
    hnsw_params = create_hnsw_params(dim, num_labels * per_label, VecSimMetric_L2,
                                     VecSimType_FLOAT32, is_multi=True)
    index = Tiered_HNSWIndex(hnsw_params, create_tiered_hnsw_params(), num_labels * per_label)

    data = np.float32(np.random.random((num_labels, per_label, dim)))
    for label in range(num_labels):
        for vector in data[label]:
            index.add_vector(vector, label)

    # In a multi index a label can hold several pending ingest jobs at once, so a late label
    # exercises re-keying all of them together while an early one is most likely already in HNSW.
    buffered = index.get_curr_bf_size()
    test_logger.info(f"relabeling with {buffered} of {num_labels * per_label} vectors buffered")
    moved = {7: num_labels + 500, num_labels - 1: num_labels + 501}
    for old_label, new_label in moved.items():
        assert index.relabel_vector(old_label, new_label) == VecSimRelabel_OK

    index.wait_for_index()

    assert index.index_size() == num_labels * per_label
    assert index.hnsw_label_count() == num_labels
    for old_label, new_label in moved.items():
        assert index.get_vector(new_label).shape == (per_label, dim)
        assert index.get_vector(old_label).shape == (0, dim)
    test_logger.info("tiered multi relabel_vector moved every vector under the label")


def test_get_vector(test_logger):
    dim = 16
    num_elements = 1000
    hnsw_params = create_hnsw_params(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32)
    # A flat buffer large enough to hold everything, so the first read below has a real chance of
    # landing while the vector is still buffered with a pending ingest job.
    index = Tiered_HNSWIndex(hnsw_params, create_tiered_hnsw_params(), num_elements)

    data = np.float32(np.random.random((num_elements, dim)))
    for label, vector in enumerate(data):
        index.add_vector(vector, label)

    # A tiered index answers from the buffer as well as from the backend, so a vector is readable
    # whichever tier currently holds it. The workers ingest in insertion order, which makes the
    # early label the likely backend case and the late one the likely buffered case.
    buffered = index.get_curr_bf_size()
    test_logger.info(f"reading vectors back with {buffered} of {num_elements} still buffered")
    for label in (0, num_elements - 1):
        assert_allclose(index.get_vector(label)[0], data[label], rtol=1e-6)

    index.wait_for_index()

    # Once ingestion has drained every vector is in HNSW, and reading it back still returns the
    # values that were inserted.
    for label in (0, 7, num_elements - 1):
        assert_allclose(index.get_vector(label)[0], data[label], rtol=1e-6)

    # An absent label is reported as no vectors rather than as an error.
    assert index.get_vector(num_elements + 1).shape == (0, dim)
    test_logger.info("tiered get_vector read from both tiers")


def test_get_vector_multi(test_logger):
    dim = 16
    num_labels = 200
    per_label = 5
    hnsw_params = create_hnsw_params(dim, num_labels * per_label, VecSimMetric_L2,
                                     VecSimType_FLOAT32, is_multi=True)
    index = Tiered_HNSWIndex(hnsw_params, create_tiered_hnsw_params(), num_labels * per_label)

    data = np.float32(np.random.random((num_labels, per_label, dim)))
    for label in range(num_labels):
        for vector in data[label]:
            index.add_vector(vector, label)

    # A multi label's vectors are routinely split across the tiers while an ingest is pending, and
    # an ingest job inserts into the backend before removing from the buffer, so a vector caught
    # inside that window is reported by both tiers. Hence the count here is a range: every returned
    # row has to be one of the label's vectors, and none of them may be missing.
    buffered = index.get_curr_bf_size()
    test_logger.info(f"reading vectors back with {buffered} of {num_labels} labels buffered")
    for label in (0, num_labels - 1):
        stored = index.get_vector(label)
        assert per_label <= stored.shape[0] <= 2 * per_label
        assert stored.shape[1] == dim
        for row in stored:
            assert np.any(np.all(np.isclose(data[label], row, rtol=1e-6), axis=1))

    index.wait_for_index()

    # Draining ingestion resolves the duplicates: each vector is in the backend exactly once.
    for label in (0, 7, num_labels - 1):
        stored = index.get_vector(label)
        assert stored.shape == (per_label, dim)
        for row in stored:
            assert np.any(np.all(np.isclose(data[label], row, rtol=1e-6), axis=1))

    assert index.get_vector(num_labels + 1).shape == (0, dim)
    test_logger.info("tiered multi get_vector read every vector under the label")


# `update_vectors` sets a label's contents: whatever it holds is removed and the given vectors take
# its place. Compared here against the add API, which is the only other way to reach that state -
# and for a single-value label the equivalent one, since adding an existing label overwrites it.
# The point of the comparison is that an index brought to a state by updating must be as good as
# one that was given that state to begin with.
def test_update_vectors(test_logger):
    indices_ctx = IndexCtx(data_size=2000, ef_r=30)
    num_labels = indices_ctx.num_labels
    dim = indices_ctx.dim
    final = indices_ctx.rng.random((num_labels, dim)).astype(indices_ctx.data.dtype)

    index = indices_ctx.tiered_index
    for i, vector in enumerate(indices_ctx.data):
        index.add_vector(vector, i)

    # Updated while the ingestion of the first vectors is still under way, so some labels are
    # replaced in the flat buffer and some in the graph.
    buffered = index.get_curr_bf_size()
    test_logger.info(f"updating {num_labels} labels, {buffered} of them still buffered")
    for i, vector in enumerate(final):
        assert index.update_vectors(i, vector) == VecSimUpdate_OK
    index.wait_for_index()

    # The same final state, reached by giving a fresh index the final vectors.
    direct = HNSWIndex(indices_ctx.hnsw_params)
    for i, vector in enumerate(final):
        direct.add_vector(vector, i)

    # No label was gained or lost on the way.
    assert index.hnsw_label_count() == num_labels
    assert direct.index_size() == num_labels

    # Every label holds what the update put there - the same stored form the add API produced for
    # it - and is found at that vector rather than at the one it used to hold.
    for i in range(0, num_labels, 50):
        assert np.allclose(index.get_vector(i), direct.get_vector(i))
        labels, distances = index.knn_query(np.array([final[i]]), 1)
        assert labels[0][0] == i
        assert distances[0][0] == pytest.approx(0, abs=1e-6)

    # On random queries the updated index is as good as the one built directly. Recall is measured
    # against the exact answer rather than compared between the two graphs, so the approximation
    # difference that parallel ingestion leaves behind cannot fail this - while a vector lost or
    # left stale by an update can.
    queries = indices_ctx.generate_queries(num_queries=10)
    k = 10
    vectors = list(enumerate(final))
    correct_updated = correct_direct = 0
    for query in queries:
        _, keys = get_ground_truth_results(spatial.distance.cosine, query, vectors, k)
        updated_labels, _ = index.knn_query(query, k)
        direct_labels, _ = direct.knn_query(query, k)
        correct_updated += len(set(updated_labels[0]) & set(keys))
        correct_direct += len(set(direct_labels[0]) & set(keys))
    recall_updated = correct_updated / (k * len(queries))
    recall_direct = correct_direct / (k * len(queries))
    test_logger.info(f"recall after updating = {recall_updated}, built with add_vector = {recall_direct}")
    assert recall_updated >= 0.9
    assert recall_updated >= recall_direct - 0.05

    # Two vectors under one label is not a state a single-value index can hold, so it refuses
    # rather than storing one of them and leaving the caller to believe both are there.
    assert index.update_vectors(0, np.array([final[0], final[1]])) == VecSimUpdate_MultiNotSupported
    assert index.hnsw_label_count() == num_labels


# For a multi-value label the update is the only way to replace the contents: `add_vector` appends,
# so a caller would have to delete the label first - and know how many vectors were there.
def test_update_vectors_multi(test_logger):
    num_per_label = 3
    indices_ctx = IndexCtx(data_size=1200, num_per_label=num_per_label, is_multi=True, ef_r=30)
    num_labels = indices_ctx.num_labels
    dim = indices_ctx.dim
    index = indices_ctx.tiered_index

    indices_ctx.populate_index(index)
    index.wait_for_index()
    assert index.hnsw_label_count() == num_labels
    assert index.index_size() == num_labels * num_per_label

    # Three vectors replaced by two: how many the label holds afterwards is decided by the update.
    new_per_label = 2
    final = indices_ctx.rng.random((num_labels, new_per_label, dim)).astype(indices_ctx.data.dtype)
    for i, vectors in enumerate(final):
        assert index.update_vectors(i, vectors) == VecSimUpdate_OK
    index.wait_for_index()

    assert index.hnsw_label_count() == num_labels
    for i in range(0, num_labels, 50):
        assert index.get_vector(i).shape == (new_per_label, dim)
        # Every one of the label's new vectors finds it, and none of the replaced ones does.
        for vector in final[i]:
            labels, distances = index.knn_query(np.array([vector]), 1)
            assert labels[0][0] == i
            assert distances[0][0] == pytest.approx(0, abs=1e-6)
        for vector in indices_ctx.data[i]:
            labels, distances = index.knn_query(np.array([vector]), 1)
            assert not (labels[0][0] == i and distances[0][0] == pytest.approx(0, abs=1e-6))

    # An empty update leaves the label holding nothing, which is a delete.
    assert index.update_vectors(0, np.empty((0, dim), dtype=indices_ctx.data.dtype)) == VecSimUpdate_OK
    index.wait_for_index()
    assert index.hnsw_label_count() == num_labels - 1
    assert index.get_vector(0).shape == (0, dim)
