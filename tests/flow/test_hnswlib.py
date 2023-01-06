# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import concurrent
import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from common import *
import hnswlib


# Helper function for creating an index,uses the default HNSW parameters if not specified.
def create_hnsw_index(dim, num_elements, metric, data_type, ef_construction=200, m=16, ef_runtime=10, epsilon=0.01,
                      is_multi=False):
    hnsw_params = HNSWParams()

    hnsw_params.dim = dim
    hnsw_params.metric = metric
    hnsw_params.type = data_type
    hnsw_params.M = m
    hnsw_params.efConstruction = ef_construction
    hnsw_params.initialCapacity = num_elements
    hnsw_params.efRuntime = ef_runtime
    hnsw_params.epsilon = epsilon
    hnsw_params.multi = is_multi

    return HNSWIndex(hnsw_params)

# compare results with the original version of hnswlib - do not use elements deletion.
def test_sanity_hnswlib_index_L2():
    dim = 16
    num_elements = 10000
    space = 'l2'
    M = 16
    efConstruction = 100
    efRuntime = 10

    index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, efConstruction, M, efRuntime)

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    data = np.float32(np.random.random((num_elements, dim)))
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    query_data = np.float32(np.random.random((1, dim)))
    hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
    redis_labels, redis_distances = index.knn_query(query_data, 10)
    assert_allclose(hnswlib_labels, redis_labels, rtol=1e-5, atol=0)
    assert_allclose(hnswlib_distances, redis_distances, rtol=1e-5, atol=0)


def test_sanity_hnswlib_index_cosine():
    dim = 16
    num_elements = 10000
    space = 'cosine'
    M = 16
    efConstruction = 100
    efRuntime = 10

    index = create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M, efRuntime)

    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    data = np.float32(np.random.random((num_elements, dim)))
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    query_data = np.float32(np.random.random((1, dim)))
    hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
    redis_labels, redis_distances = index.knn_query(query_data, 10)
    assert_allclose(hnswlib_labels, redis_labels, rtol=1e-5, atol=0)
    assert_allclose(hnswlib_distances, redis_distances, rtol=1e-5, atol=0)


# Validate correctness of delete implementation comparing the brute force search. We test the search recall which is not
# deterministic, but should be above a certain threshold. Note that recall is highly impacted by changing
# index parameters.
def test_recall_for_hnswlib_index_with_deletion():
    dim = 16
    num_elements = 10000
    M = 16
    efConstruction = 100

    num_queries = 10
    k = 10
    efRuntime = 0

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, efConstruction, M, efRuntime)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    # delete half of the data
    for i in range(0, len(data), 2):
        hnsw_index.delete_vector(i)
    vectors = [vectors[i] for i in range(1, len(data), 2)]

    # We validate that we can increase ef with this designated API (if this won't work, recall should be very low)
    hnsw_index.set_ef(50)
    query_data = np.float32(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        keys = [key for _, key in dists[:k]]

        for label in hnswlib_labels[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct += 1
                    break

    # Measure recall
    recall = float(correct) / (k * num_queries)
    print("\nrecall is: \n", recall)
    assert (recall > 0.9)


def test_batch_iterator():
    dim = 100
    num_elements = 100000
    M = 26
    efConstruction = 180
    efRuntime = 180
    num_queries = 10

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, efConstruction, M, efRuntime)

    # Add 100k random vectors to the index
    rng = np.random.default_rng(seed=47)
    data = np.float32(rng.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    # Create a random query vector and create a batch iterator
    query_data = np.float32(rng.random((1, dim)))
    batch_iterator = hnsw_index.create_batch_iterator(query_data)
    labels_first_batch, distances_first_batch = batch_iterator.get_next_results(10, BY_ID)
    for i, _ in enumerate(labels_first_batch[0][:-1]):
        # Assert sorting by id
        assert (labels_first_batch[0][i] < labels_first_batch[0][i + 1])

    labels_second_batch, distances_second_batch = batch_iterator.get_next_results(10, BY_SCORE)
    should_have_return_in_first_batch = []
    for i, dist in enumerate(distances_second_batch[0][:-1]):
        # Assert sorting by score
        assert (distances_second_batch[0][i] < distances_second_batch[0][i + 1])
        # Assert that every distance in the second batch is higher than any distance of the first batch
        if len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) != 0:
            should_have_return_in_first_batch.append(dist)
    assert (len(should_have_return_in_first_batch) <= 2)

    # Verify that runtime args are sent properly to the batch iterator.
    query_params = VecSimQueryParams()
    query_params.hnswRuntimeParams.efRuntime = 5
    batch_iterator_new = hnsw_index.create_batch_iterator(query_data, query_params)
    labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
    # Verify that accuracy is worse with the new lower ef_runtime.
    assert (sum(labels_first_batch[0]) < sum(labels_first_batch_new[0]))

    query_params.hnswRuntimeParams.efRuntime = efRuntime  # Restore previous ef_runtime.
    batch_iterator_new = hnsw_index.create_batch_iterator(query_data, query_params)
    labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
    # Verify that results are now the same.
    assert_allclose(labels_first_batch_new[0], labels_first_batch[0])

    # Reset
    batch_iterator.reset()

    # Run in batches of 100 until we reach 1000 results and measure recall
    batch_size = 100
    total_res = 1000
    total_recall = 0
    query_data = np.float32(rng.random((num_queries, dim)))
    for target_vector in query_data:
        correct = 0
        batch_iterator = hnsw_index.create_batch_iterator(target_vector)
        iterations = 0
        # Sort distances of every vector from the target vector and get the actual order
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
          total_recall / num_queries)

    # Run again a single query in batches until it is depleted.
    batch_iterator = hnsw_index.create_batch_iterator(query_data[0])
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


def test_serialization():
    dim = 16
    num_elements = 10000
    M = 16
    efConstruction = 100
    data_type = VecSimType_FLOAT32

    num_queries = 10
    k = 10
    efRuntime = 50

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, data_type, efConstruction, M, efRuntime)
    hnsw_index.set_ef(efRuntime)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((num_queries, dim)))
    correct = 0
    correct_labels = []  # cache these
    for target_vector in query_data:
        hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        keys = [key for _, key in dists[:k]]
        correct_labels.append(keys)

        for label in hnswlib_labels[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct += 1
                    break
    # Measure recall
    recall = float(correct) / (k * num_queries)
    print("\nrecall is: \n", recall)

    # Persist, delete and restore index.
    file_name = os.getcwd() + "/dump"
    hnsw_index.save_index(file_name)

    new_hnsw_index = HNSWIndex(file_name)
    os.remove(file_name)
    assert new_hnsw_index.index_size() == num_elements

    # Check recall
    correct_after = 0
    for i, target_vector in enumerate(query_data):
        hnswlib_labels, hnswlib_distances = new_hnsw_index.knn_query(target_vector, 10)
        correct_labels_cur = correct_labels[i]
        for label in hnswlib_labels[0]:
            for correct_label in correct_labels_cur:
                if label == correct_label:
                    correct_after += 1
                    break

    # Compare recall after reloading the index
    recall_after = float(correct_after) / (k * num_queries)
    print("\nrecall after is: \n", recall_after)
    assert recall == recall_after


def test_range_query():
    dim = 100
    num_elements = 100000
    epsilon = 0.01

    index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, ef_construction=200, m=32,
                                   epsilon=epsilon)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))

    radius = 13.0
    recalls = {}

    for epsilon_rt in [0.001, 0.01, 0.1]:
        query_params = VecSimQueryParams()
        query_params.hnswRuntimeParams.epsilon = epsilon_rt
        start = time.time()
        hnsw_labels, hnsw_distances = index.range_query(query_data, radius=radius, query_param=query_params)
        end = time.time()
        res_num = len(hnsw_labels[0])

        dists = sorted([(key, spatial.distance.sqeuclidean(query_data.flat, vec)) for key, vec in vectors])
        actual_results = [(key, dist) for key, dist in dists if dist <= radius]

        print(
            f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
            f' got {res_num} results, which are {res_num / len(actual_results)} of the entire results in the range.')

        # Compare the number of vectors that are actually within the range to the returned results.
        assert np.all(np.isin(hnsw_labels, np.array([label for label, _ in actual_results])))

        assert max(hnsw_distances[0]) <= radius
        recalls[epsilon_rt] = res_num / len(actual_results)

    # Expect higher recalls for higher epsilon values.
    assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

    # Expect zero results for radius==0
    hnsw_labels, hnsw_distances = index.range_query(query_data, radius=0)
    assert len(hnsw_labels[0]) == 0


def test_recall_for_hnsw_multi_value():
    dim = 16
    num_labels = 1000
    num_per_label = 16
    M = 16
    efConstruction = 100
    num_queries = 10
    k = 10
    efRuntime = 0

    num_elements = num_labels * num_per_label

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, efConstruction, M,
                                   efRuntime, is_multi=True)

    data = np.float32(np.random.random((num_labels, dim)))
    vectors = []
    for i, vector in enumerate(data):
        for _ in range(num_per_label):
            hnsw_index.add_vector(vector, i)
            vectors.append((i, vector))

    # We validate that we can increase ef with this designated API (if this won't work, recall should be very low)
    hnsw_index.set_ef(50)
    query_data = np.float32(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)
        assert (len(hnswlib_labels[0]) == len(np.unique(hnswlib_labels[0])))

        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = {}
        for key, vec in vectors:
            # Setting or updating the score for each label. If it's the first time we calculate a score for a label,
            # dists.get(key, 3) will return 3, which is more than a Cosine score can be,
            # so we will choose the actual score the first time.
            dists[key] = min(spatial.distance.cosine(target_vector, vec),
                             dists.get(key, 3))  # cosine distance is always <= 2

        dists = list(dists.items())
        dists = sorted(dists, key=lambda pair: pair[1])[:k]
        keys = [key for key, _ in dists]

        for label in hnswlib_labels[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct += 1
                    break

    # Measure recall
    recall = float(correct) / (k * num_queries)
    print("\nrecall is: \n", recall)
    assert (recall > 0.9)


def test_multi_range_query():
    dim = 100
    num_labels = 20000
    per_label = 5
    epsilon = 0.01
    num_elements = num_labels * per_label

    index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, ef_construction=200, m=32,
                              epsilon=epsilon, is_multi=True)

    np.random.seed(47)
    data = np.float32(np.random.random((num_labels, per_label, dim)))
    vectors = []
    for label, vecs in enumerate(data):
        for vector in vecs:
            index.add_vector(vector, label)
            vectors.append((label, vector))

    query_data = np.float32(np.random.random((1, dim)))

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
        hnsw_labels, hnsw_distances = index.range_query(query_data, radius=radius, query_param=query_params)
        end = time.time()
        res_num = len(hnsw_labels[0])

        print(
            f'\nlookup time for ({num_labels} X {per_label}) vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
            f' got {res_num} results, which are {res_num / len(keys)} of the entire results in the range.')

        # Compare the number of vectors that are actually within the range to the returned results.
        assert np.all(np.isin(hnsw_labels, np.array(keys)))

        # Asserts that all the results are unique
        assert len(hnsw_labels[0]) == len(np.unique(hnsw_labels[0]))

        assert max(hnsw_distances[0]) <= radius
        recalls[epsilon_rt] = res_num / len(keys)

    # Expect higher recalls for higher epsilon values.
    assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

    # Expect zero results for radius==0
    hnsw_labels, hnsw_distances = index.range_query(query_data, radius=0)
    assert len(hnsw_labels[0]) == 0


def test_parallel_insert_search():
    dim = 32
    num_elements = 100000
    num_queries = 10000
    k = 10
    n_threads = int(os.cpu_count() / 2)
    expected_parallel_rate = 0.9  # we expect that at least 90% of the insert/search time will be executed in parallel
    expected_speedup = 1 / ((1-expected_parallel_rate) + expected_parallel_rate/n_threads)  # by Amdahl's law

    # Create two HNSW indexes, one for sequential insertion and one for parallel insertion of vectors.
    index = create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, ef_runtime=200)
    parallel_index = create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, ef_runtime=200)

    bf_params = BFParams()

    bf_params.initialCapacity = num_elements
    bf_params.blockSize = num_elements
    bf_params.dim = dim
    bf_params.type = VecSimType_FLOAT32
    bf_params.metric = VecSimMetric_Cosine

    bf_index = BFIndex(bf_params)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    start = time.time()
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
    sequential_insert_time = time.time() - start
    print(f"Inserting {num_elements} vectors of dim {dim} into HNSW sequentially took {sequential_insert_time} seconds")

    start = time.time()
    parallel_index.add_vector_parallel(data, np.array(range(num_elements)), n_threads)
    parallel_insert_time = time.time() - start
    assert parallel_index.index_size() == num_elements
    assert parallel_index.check_integrity()
    # Validate that the parallel index contains the same vectors as the sequential one.
    for label in range(num_elements):
        assert_allclose(index.get_vector(label), parallel_index.get_vector(label))

    print(f"Inserting {num_elements} vectors of dim {dim} into HNSW in parallel took {parallel_insert_time} seconds")
    print(f"Got {sequential_insert_time/parallel_insert_time} times improvement using {n_threads} threads\n")
    assert sequential_insert_time/parallel_insert_time > expected_speedup

    for i, vector in enumerate(data):
        bf_index.add_vector(vector, i)

    query_data = np.float32(np.random.random((num_queries, dim)))

    # Sequential search as the baseline
    total_search_time = 0
    total_correct = 0
    total_res_bf = []  # save the ground truth
    for i, query in enumerate(query_data):
        start = time.time()
        res_labels, _ = index.knn_query(query, k)
        total_search_time += time.time() - start
        res_labels_bf, _ = bf_index.knn_query(query, k)
        total_res_bf.append(set(res_labels_bf[0]))
        total_correct += len(set(res_labels[0]).intersection(set(res_labels_bf[0])))

    print(f"Running sequential search, got {total_correct / (k * num_queries)} recall on {num_queries} queries,"
          f" average query time is {total_search_time / num_queries} seconds")

    start = time.time()
    res_labels, _ = index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time_parallel = time.time() - start

    total_correct_parallel = 0
    for i in range(num_queries):
        total_correct_parallel += len(total_res_bf[i].intersection(set(res_labels[i])))

    print(f"Running parallel search, got {total_correct_parallel / (k * num_queries)} recall on {num_queries} queries,"
          f" average query time is {total_search_time_parallel / num_queries} seconds")
    print(f"Got {total_search_time / total_search_time_parallel} times improvement un runtime using {n_threads} threads\n")

    # Validate that the recall of the parallel search recall is the same as the sequential search recall.
    assert total_correct_parallel == total_correct
    # Validate that the parallel run managed to achieve at least (n_threads - 1) times improvement in total runtime.
    assert total_search_time / total_search_time_parallel > expected_speedup

    # Run search with parallel index and assert that similar recall achieved.
    start = time.time()
    res_labels, _ = parallel_index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time_parallel = time.time() - start

    total_correct_parallel = 0
    for i in range(num_queries):
        total_correct_parallel += len(total_res_bf[i].intersection(set(res_labels[i])))
    print(f"Running parallel search on index that was created using parallel insert, got "
          f"{total_correct_parallel / (k * num_queries)} recall on {num_queries} queries, average query time is"
          f" {total_search_time_parallel / num_queries} seconds")
    assert total_correct_parallel >= total_correct * 0.95

    # Insert vectors to the index and search in parallel.
    parallel_index = create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, ef_runtime=200)
    assert parallel_index.index_size() == 0

    def insert_vectors():
        parallel_index.add_vector_parallel(data, np.array(range(num_elements)), num_threads=int(n_threads/2))

    res_labels_g = np.zeros((num_queries, dim))

    def run_queries():
        nonlocal res_labels_g
        res_labels_g, _ = parallel_index.knn_parallel(query_data, k, num_threads=int(n_threads/2))

    t_insert = threading.Thread(target=insert_vectors)
    t_query = threading.Thread(target=run_queries)
    print("Running queries in parallel to inserting vectors to the index, start running queries after more 50% of the"
          " vectors are indexed")
    t_insert.start()
    # Wait until half of the index is indexed, then start run queries
    while parallel_index.index_size() < num_elements / 2:
        time.sleep(0.5)
    t_query.start()

    [t.join() for t in [t_insert, t_query]]

    # Measure recall - expect to get increased recall over time, since vectors are being inserted while queries
    # are running, and the ground truth is measured compared to the index that contains all the elements.
    chunk_size = int(num_queries/5)
    total_correct_prev_chunk = 0
    for i in range(0, num_queries, chunk_size):
        total_correct_cur_chunk = 0
        for j in range(i, i+chunk_size):
            total_correct_cur_chunk += len(total_res_bf[j].intersection(set(res_labels_g[j])))
        if i == chunk_size:  # first iteration, there is no previous chunk
            total_correct_prev_chunk = total_correct_cur_chunk
        else:
            assert total_correct_cur_chunk >= total_correct_prev_chunk
            total_correct_prev_chunk = total_correct_cur_chunk
        print(f"Recall for chunk {int(i/chunk_size)+1}/{int(num_queries/chunk_size)} of queries is:"
              f" {total_correct_cur_chunk/(k*chunk_size)}")


def test_parallel_with_range():
    dim = 32
    num_elements = 100000
    num_queries = 10000
    radius = 3.0
    n_threads = int(os.cpu_count() / 2)
    PADDING_LABEL = -1  # used for padding empty labels entries in a single query results
    expected_parallel_rate = 0.9  # we expect that at least 90% of the insert/search time will be executed in parallel
    expected_speedup = 1 / ((1-expected_parallel_rate) + expected_parallel_rate/n_threads)  # by Amdahl's law

    parallel_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT32, ef_runtime=200)

    bf_params = BFParams()

    bf_params.dim = dim
    bf_params.metric = VecSimMetric_L2
    bf_params.type = VecSimType_FLOAT32
    bf_params.initialCapacity = num_elements
    bf_params.blockSize = num_elements

    bf_index = BFIndex(bf_params)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    for i, vector in enumerate(data):
        bf_index.add_vector(vector, i)

    parallel_index.add_vector_parallel(data, range(num_elements), n_threads)
    query_data = np.float32(np.random.random((num_queries, dim)))

    # Run serial range queries
    total_search_time = 0
    total_res_bf = []  # ground truth
    # The ratio between then number of results returned by HNSW and the total number of vectors in the range.
    overall_intersection_rate = 0
    total_results = 0
    for i, query in enumerate(query_data):
        start = time.time()
        res_labels_range, res_distances_range = parallel_index.range_query(query, radius)
        total_search_time += time.time() - start
        res_labels_bf_range, res_distances_bf_range = bf_index.range_query(query, radius)
        assert set(res_labels_range[0]).issubset(set(res_labels_bf_range[0]))
        total_res_bf.append(res_labels_bf_range[0])
        total_results += res_labels_bf_range[0].size
        overall_intersection_rate += res_labels_range[0].size / res_labels_bf_range[0].size\
            if res_labels_bf_range[0].size > 0 else 1
    print(f"Range queries - running {num_queries} queries sequentially, average number of results is:"
          f" {total_results/num_queries} and HNSW success rate is: {overall_intersection_rate/num_queries}."
          f" Average query time is {total_search_time/num_queries} seconds")

    # Run range queries in parallel
    start = time.time()
    hnsw_labels_range_parallel, _ = parallel_index.range_parallel(query_data, radius=radius)
    total_range_query_parallel_time = time.time() - start
    overall_intersection_rate_parallel = 0
    for i in range(num_queries):
        query_results_set = set(hnsw_labels_range_parallel[i])
        query_results_set.discard(PADDING_LABEL)  # remove the irrelevant padding values
        assert query_results_set.issubset(set(total_res_bf[i]))
        overall_intersection_rate_parallel += len(query_results_set) / total_res_bf[i].size \
            if total_res_bf[i].size > 0 else 1
    print(f"Running the same {num_queries} queries in parallel, average query time is"
          f" {total_range_query_parallel_time/num_queries}, and intersection rate is: "
          f"{overall_intersection_rate_parallel/num_queries}")
    assert overall_intersection_rate_parallel == overall_intersection_rate
    print(f"Got improvement of {total_search_time/total_range_query_parallel_time} times using {n_threads} threads\n")
    assert total_search_time/total_range_query_parallel_time >= expected_speedup


def test_parallel_with_multi():
    dim = 32
    num_labels = 10000
    per_label = 5
    epsilon = 0.01
    k = 10
    num_elements = num_labels * per_label
    metric = VecSimMetric_L2
    data_type = VecSimType_FLOAT64
    num_queries = 10000
    n_threads = int(os.cpu_count() / 2)
    expected_parallel_rate = 0.9  # we expect that at least 90% of the insert/search time will be executed in parallel
    expected_speedup = 1 / ((1-expected_parallel_rate) + expected_parallel_rate/n_threads)  # by Amdahl's law

    multi_index = create_hnsw_index(dim, num_elements, metric, data_type, m=32, ef_runtime=200,
                                    epsilon=epsilon, is_multi=True)
    parallel_multi_index = create_hnsw_index(dim, num_elements, metric, data_type, m=32, ef_runtime=200,
                                             epsilon=epsilon, is_multi=True)
    bf_params = BFParams()

    bf_params.dim = dim
    bf_params.metric = metric
    bf_params.type = data_type
    bf_params.initialCapacity = num_elements
    bf_params.blockSize = num_elements
    bf_params.multi = True

    bf_index = BFIndex(bf_params)

    np.random.seed(47)
    data = np.random.random((num_labels, per_label, dim))
    sequential_insert_time = 0
    for label, vectors in enumerate(data):
        for vector in vectors:
            start = time.time()
            multi_index.add_vector(vector, label)
            sequential_insert_time += time.time() - start
            bf_index.add_vector(vector, label)
    print(f"Inserting {num_elements} vectors of dim {dim} into multi-HNSW ({per_label} vectors per label) sequentially"
          f" took {sequential_insert_time} seconds")

    # Insert vectors to multi index in parallel
    data = data.reshape(num_elements, dim)
    labels = np.concatenate([[i]*per_label for i in range(num_labels)])
    start = time.time()
    parallel_multi_index.add_vector_parallel(data, labels, n_threads)
    parallel_insert_time = time.time() - start
    assert parallel_multi_index.index_size() == num_elements
    assert parallel_multi_index.check_integrity()
    # Validate that the parallel index contains the same vectors as the sequential one. vectors are not necessarily
    # at the same order, so we flatten the array and check that elements are set equal.
    # x=input("now")
    for label in range(num_labels):
        vectors_s = multi_index.get_vector(label)
        vectors_p = parallel_multi_index.get_vector(label)
        assert vectors_s.shape == vectors_p.shape
        assert set(vectors_s.flatten()) == set(vectors_p.flatten())

    print(f"Inserting {num_elements} vectors of dim {dim} into multi-HNSW in parallel ({per_label} vectors per label)"
          f" took {parallel_insert_time} seconds")
    print(f"Got {sequential_insert_time/parallel_insert_time} times improvement using {n_threads} threads\n")
    assert sequential_insert_time/parallel_insert_time > expected_speedup

    # Run queries over the multi-index
    query_data = np.random.random((num_queries, dim))

    # Sequential search as the baseline
    total_search_time = 0
    total_correct = 0
    total_res_bf = []  # save the ground truth res_labels for every query
    total_res_hnsw_sequential = []  # save sequential search results, tuple of (res_labels, res_dists) for every query
    for i, query in enumerate(query_data):
        start = time.time()
        res_labels, res_dists = multi_index.knn_query(query, k)
        total_search_time += time.time() - start
        res_labels_bf, res_dists_bf = bf_index.knn_query(query, k)
        total_res_hnsw_sequential.append((res_labels[0], res_dists[0]))
        total_res_bf.append(res_labels_bf[0])
        total_correct += len(set(res_labels[0]).intersection(set(res_labels_bf[0])))
    print(f"Running sequential search, got {total_correct / (k * num_queries)} recall on {num_queries} queries,"
          f" average query time is {total_search_time / num_queries} seconds")

    start = time.time()
    res_labels_parallel, res_dists_parallel = parallel_multi_index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time_parallel = time.time() - start

    # # Validate that we got the same results as in the sequential index
    # for res_labels, res_dists, sequential_res in zip(res_labels_parallel, res_dists_parallel, total_res_hnsw_sequential):
    #     assert_allclose(res_dists, sequential_res[1])
    #     assert set(res_labels) == set(sequential_res[0])

    total_correct_parallel = 0
    for res_labels, ground_truth in zip(res_labels_parallel, total_res_bf):
        total_correct_parallel += len(set(res_labels).intersection(set(ground_truth)))

    print(f"Running parallel search, got {total_correct_parallel / (k * num_queries)} recall on {num_queries} queries,"
          f" average query time is {total_search_time_parallel / num_queries} seconds")
    print(f"Got {total_search_time / total_search_time_parallel} times improvement un runtime using"
          f" {n_threads} threads\n")
    assert total_correct_parallel >= total_correct*0.95
    assert total_search_time/total_search_time_parallel >= expected_speedup

    # todo: cont testing parallel insert+search
