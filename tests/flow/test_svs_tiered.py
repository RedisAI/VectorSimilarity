# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).
import time
import pytest
from common import *

# trainingThreshold = 0 means use the default trainingTriggerThreshold defined in svs_tiered.h
# updateThreshold = 0 means use the default updateTriggerThreshold defined in svs_tiered.h
def create_tiered_svs_params(trainingThreshold = 0, updateThreshold = 0):
    tiered_svs_params = TieredSVSParams()
    tiered_svs_params.trainingTriggerThreshold = trainingThreshold
    tiered_svs_params.updateTriggerThreshold = updateThreshold
    tiered_svs_params.updateJobWaitTime = 0
    return tiered_svs_params

class IndexCtx:
    array_conversion_func = {
        VecSimType_FLOAT32: np.float32,
        VecSimType_FLOAT16: vec_to_float16,
    }

    type_to_dtype = {
        VecSimType_FLOAT32: np.float32,
        VecSimType_FLOAT16: np.float16,
    }

    def __init__(self, data_size=10000,
                 dim=16,
                 graph_degree=64,
                 window_size_c=128,
                 window_size_r=20,
                 max_candidate_pool_size = 0,
                 prune_to = 0,
                 metric=VecSimMetric_Cosine,
                 data_type=VecSimType_FLOAT32,
                 quantBits=VecSimSvsQuant_NONE,
                 is_multi=False,
                 num_per_label=1,
                 trainingThreshold=1024,
                 updateThreshold=16,
                 flat_buffer_size=1024,
                 num_threads=0,
                 create_data_func = None):
        assert not is_multi, "Multi-label tiered index is not supported yet"
        self.num_vectors = data_size
        self.dim = dim
        self.graph_degree = graph_degree
        self.window_size_c = window_size_c
        self.window_size_r = window_size_r
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

        self.svs_params = create_svs_params(dim = self.dim,
                                            num_elements = self.num_vectors,
                                            metric = self.metric,
                                            data_type = self.data_type,
                                            quantBits= quantBits,
                                            graph_max_degree = self.graph_degree,
                                            construction_window_size = self.window_size_c,
                                            search_window_size = self.window_size_r,
                                            max_candidate_pool_size = max_candidate_pool_size,
                                            prune_to = prune_to,
                                            num_threads=num_threads)
        self.tiered_svs_params = create_tiered_svs_params(trainingThreshold, updateThreshold)

        self.tiered_index = Tiered_SVSIndex(self.svs_params, self.tiered_svs_params, flat_buffer_size)

    # TODO - add multi-label support
    # def populate_index_multi(self, index):
    #     start = time.time()
    #     duration = 0
    #     for label, vectors in enumerate(self.data):
    #         for vector in vectors:
    #             start_add = time.time()
    #             index.add_vector(vector, label)
    #             duration += time.time() - start_add
    #     end = time.time()
    #     return (start, duration, end)

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

    def create_svs_index(self):
        return SVSIndex(self.svs_params)

    def init_and_populate_svs_index(self):
        svs_index = SVSIndex(self.svs_params)
        self.svs_index = svs_index

        self.populate_index(svs_index)
        return svs_index

    def generate_queries(self, num_queries):
        queries = self.create_data_func((num_queries, self.dim))
        if self.data_type in self.array_conversion_func.keys():
            queries = self.array_conversion_func[self.data_type](queries)
        return queries

    def get_vectors_memory_size(self):
        memory_size = {
            VecSimType_FLOAT32: 4,
            VecSimType_FLOAT16: 2,
        }
        return bytes_to_mega(self.num_vectors * self.dim * memory_size[self.data_type])

def create_tiered_index(test_logger, is_multi: bool, num_per_label=1, data_type=VecSimType_FLOAT32, quantBits=VecSimSvsQuant_NONE, create_data_func=None):
    indices_ctx = IndexCtx(data_size=50000, is_multi=is_multi, num_per_label=num_per_label, data_type=data_type,
                           quantBits=quantBits, create_data_func=create_data_func)
    test_logger.info(f"data type = {indices_ctx.data.dtype}")
    num_elements = indices_ctx.num_labels

    index = indices_ctx.tiered_index
    threads_num = index.get_threads_num()

    _, bf_dur, end_add_time = indices_ctx.populate_index(index)

    index.wait_for_index()
    tiered_index_time = bf_dur + time.time() - end_add_time

    assert index.svs_label_count() >= num_elements - indices_ctx.tiered_svs_params.updateTriggerThreshold

    # Measure insertion to tiered index.
    test_logger.info(f"Insert {num_elements} vectors into the flat buffer took {round_ms(bf_dur)} ms")
    test_logger.info(f"Total time for inserting vectors to the tiered index and indexing them into SVS using {threads_num}"
          f" threads took {round_ms(tiered_index_time)} ms")

    # Measure total memory of the tiered index.
    tiered_memory = bytes_to_mega(index.index_memory())

    test_logger.info(f"total memory of tiered index = {tiered_memory} MB")

    svs_index = SVSIndex(indices_ctx.svs_params)
    _, svs_index_time, _ = indices_ctx.populate_index(svs_index)

    test_logger.info(f"Insert {num_elements} vectors directly to SVS index (one by one) took {round_(svs_index_time)} s")
    svs_memory = bytes_to_mega(svs_index.index_memory())
    test_logger.info(f"total memory of svs index = {svs_memory} MB")

    # The index memory should be at least as the total memory of the vectors.
    assert svs_memory > indices_ctx.get_vectors_memory_size()

    # Tiered index memory should be greater than SVS index memory.
    assert tiered_memory > svs_memory
    execution_time_ratio = svs_index_time / tiered_index_time
    test_logger.info(f"with {threads_num} threads, insertion runtime is {round_(execution_time_ratio)} times better \n")

def correlated_data_func(shape):
    """
    Generates correlated data for testing purposes.
    Correlated data is required to emulate real datasets for SVS LeanVec compression.
    The data is generated such that the first half of the last dimension is correlated with the second half.
    Supports 2D or 3D shapes.
    """
    rng = np.random.default_rng(seed=47)
    data = rng.normal(1, 0.5, shape)
    # Correlate the first half with the second half along the last dimension
    # This is done by adding noise to the even indices and keeping the odd indices as is.
    # This way we ensure that the even indices are correlated with the odd indices.
    # We use a slice to select every second element in the last dimension.
    # For example, if the shape is (100, 10), we will have
    # data[:, 0] = data[:, 1] + noise, data[:, 2] = data[:, 3] + noise, etc.
    # This way we ensure that the first half of the last dimension is correlated with the second half.
    slicer_even = [slice(None)] * (len(shape) - 1) + [slice(0, None, 2)]
    slicer_odd = [slice(None)] * (len(shape) - 1) + [slice(1, None, 2)]
    data[tuple(slicer_even)] = data[tuple(slicer_odd)] + rng.normal(0, 0.1, data[tuple(slicer_even)].shape)

    return data

def search_insert(test_logger, is_multi: bool, num_per_label=1, data_type=VecSimType_FLOAT32, quantBits=VecSimSvsQuant_NONE, create_data_func=correlated_data_func):
    data_size = 100000
    trainingThreshold = 10240
    updateThreshold = 1024
    indices_ctx = IndexCtx(dim=32, data_size=data_size, is_multi=is_multi, num_per_label=num_per_label,
                           flat_buffer_size=data_size, graph_degree=32, data_type=data_type, quantBits=quantBits,
                           create_data_func=create_data_func, trainingThreshold=trainingThreshold, updateThreshold=updateThreshold)
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
    cur_svs_label_count = index.svs_label_count()

    test_logger.info(f"SVS labels number = {cur_svs_label_count}")
    while searches_number == 0 or cur_svs_label_count < num_labels - updateThreshold:
        # For each run get the current svs size and the query time.
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
        cur_svs_label_count = index.svs_label_count()

    # SVS labels count updates before the job is done, so we need to wait for the queue to be empty.
    index.wait_for_index(1)
    index_dur = time.time() - index_start
    test_logger.info(f"Indexing in the tiered index took {round_(index_dur)} s")

    # Measure recall.
    recall = float(correct)/(k*searches_number)
    test_logger.info(f"Average recall is: {round_(recall, 3)}")
    test_logger.info(f"tiered query per seconds: {round_(searches_number/total_tiered_search_time)}")


def test_create_tiered(test_logger):
    test_logger.info("Test create tiered svs index")
    create_tiered_index(test_logger, is_multi=False)

# TODO - add multi-label support
@pytest.mark.skip(reason="Multi-label tiered index is not supported yet")
def test_create_multi(test_logger):
    test_logger.info("Test create multi label tiered svs index")
    create_tiered_index(test_logger, is_multi=True, num_per_label=5)

def test_create_fp16(test_logger):
    test_logger.info("Test create FLOAT16 tiered svs index")
    create_tiered_index(test_logger, is_multi=False, data_type=VecSimType_FLOAT16)

def test_search_insert(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False)

def test_search_insert_q8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, quantBits=VecSimSvsQuant_8)

def test_search_insert_q4(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, quantBits=VecSimSvsQuant_4)

def test_search_insert_leanvec_8x8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, quantBits=VecSimSvsQuant_8x8_LeanVec)

def test_search_insert_leanvec_4x8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, quantBits=VecSimSvsQuant_4x8_LeanVec)

def test_search_insert_fp16(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_FLOAT16)

def test_search_insert_fp16_q8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_FLOAT16, quantBits=VecSimSvsQuant_8)

def test_search_insert_fp16_q4(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_FLOAT16, quantBits=VecSimSvsQuant_4)

def test_search_insert_fp16_leanvec_8x8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_FLOAT16, quantBits=VecSimSvsQuant_8x8_LeanVec)

def test_search_insert_fp16_leanvec_4x8(test_logger):
    test_logger.info("Start insert & search test")
    search_insert(test_logger, is_multi=False, data_type=VecSimType_FLOAT16, quantBits=VecSimSvsQuant_4x8_LeanVec)

# TODO - add multi-label support
@pytest.mark.skip(reason="Multi-label tiered index is not supported yet")
def test_search_insert_multi_index(test_logger):
    test_logger.info("Start insert & search test for multi index")

    search_insert(test_logger, is_multi=True, num_per_label=5)

# In this test we insert the vectors one by one to the tiered index (call wait_for_index after each add vector)
# We expect to get the same index as if we were inserting the vector to the sync svs index.
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

    assert index.svs_label_count() >= indices_ctx.num_labels - indices_ctx.tiered_svs_params.updateTriggerThreshold

    # Create svs index.
    svs_index = indices_ctx.init_and_populate_svs_index()

    query_data = indices_ctx.generate_queries(num_queries=1)

    # Search knn in tiered.
    tiered_labels, tiered_dist = index.knn_query(query_data, k)
    # Search knn in svs.
    svs_labels, svs_dist = svs_index.knn_query(query_data, k)

    # Compare.
    has_diff = False
    for i, svs_res_label in enumerate(svs_labels[0]):
        if svs_res_label != tiered_labels[0][i]:
            has_diff = True
            test_logger.info(f"svs label = {svs_res_label}, tiered label = {tiered_labels[0][i]}")
            test_logger.info(f"svs dist = {svs_dist[0][i]}, tiered dist = {tiered_dist[0][i]}")

    assert not has_diff
    test_logger.info(f"svs graph is identical to the tiered index graph")


def test_recall_after_deletion(test_logger):

    indices_ctx = IndexCtx(window_size_r=30)
    index = indices_ctx.tiered_index
    data = indices_ctx.data
    num_elements = indices_ctx.num_labels

    # Create svs index.
    svs_index = indices_ctx.init_and_populate_svs_index()

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
        svs_index.delete_vector(i)

    # Wait for all repair jobs to be done.
    index.wait_for_index(5)
    test_logger.info(f"Done deleting half of the index")
    assert index.svs_label_count() >= (num_elements // 2) - indices_ctx.tiered_svs_params.updateTriggerThreshold
    assert index.svs_label_count() <= (num_elements // 2) + indices_ctx.tiered_svs_params.updateTriggerThreshold
    assert svs_index.index_size() == (num_elements // 2)

    # Create a list of tuples of the vectors that left.
    vectors = [vectors[i] for i in range(1, num_elements, 2)]

    # Perform queries.
    num_queries = 10
    queries = indices_ctx.generate_queries(num_queries=10)

    k = 10
    correct_tiered = 0
    correct_svs = 0

    # Calculate correct vectors for each index.
    # We don't expect svs and tiered svs results to be identical due to the parallel insertion.
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
        svs_labels, _ = svs_index.knn_query(target_vector, k)

        # Sort distances of every vector from the target vector and get actual k nearest vectors.
        dists = [(spatial.distance.cosine(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        keys = [key for _, key in dists[:k]]
        correct_tiered += calculate_correct(tiered_labels, keys)
        correct_svs += calculate_correct(svs_labels, keys)

    # Measure recall.
    recall_tiered = float(correct_tiered) / (k * num_queries)
    recall_svs = float(correct_svs) / (k * num_queries)
    test_logger.info(f"SVS tiered recall is: {recall_tiered}")
    test_logger.info(f"SVS recall is: {recall_svs}")
    assert (recall_tiered >= 0.9)


def test_batch_iterator(test_logger):
    num_elements = 50000
    dim = 100
    graphDegree = 64
    wsConstruction = 200
    wsRuntime = 100
    metric = VecSimMetric_L2
    indices_ctx = IndexCtx(data_size=num_elements,
                           dim=dim,
                           graph_degree=graphDegree,
                           window_size_c=wsConstruction,
                           window_size_r=wsRuntime,
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
        assert iterations == int(np.ceil(total_res / batch_size))
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
    num_elements = 50000
    dim = 100
    wsConstruction = 200
    wsRuntime = 10
    metric = VecSimMetric_L2

    indices_ctx = IndexCtx(data_size=num_elements,
                        dim=dim,
                        window_size_c=wsConstruction,
                        window_size_r=wsRuntime,
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
        query_params.svsRuntimeParams.epsilon = epsilon_rt
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


# TODO - add multi-label support
@pytest.mark.skip(reason="Multi-label tiered index is not supported yet")
def test_multi_range_query(test_logger):
    num_labels = 20000
    per_label = 5
    num_elements = num_labels * per_label

    dim = 100
    wsConstruction = 200
    wsRuntime = 10
    metric = VecSimMetric_L2

    indices_ctx = IndexCtx(data_size=num_elements,
                        dim=dim,
                        window_size_c=wsConstruction,
                        window_size_r=wsRuntime,
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
        query_params.svsRuntimeParams.epsilon = epsilon_rt
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
