# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).
import concurrent
import math
import multiprocessing
import os
import time
from VecSim import *
from common import *
import hnswlib

def create_svs_index(dim, num_elements, data_type, metric = VecSimMetric_L2,
                     alpha = 1.2, graph_max_degree = 64, window_size = 128,
                     max_candidate_pool_size = 1024, prune_to = 60, full_search_history = VecSimOption_AUTO,
                     search_window_size = 20, epsilon = 0.01, num_threads = 4, is_multi = False):
    svs_params = SVSParams()

    svs_params.dim = dim
    svs_params.type = data_type
    svs_params.metric = metric
    svs_params.multi = is_multi
    svs_params.alpha = alpha
    svs_params.graph_max_degree = graph_max_degree
    svs_params.construction_window_size = window_size
    svs_params.max_candidate_pool_size = max_candidate_pool_size
    svs_params.prune_to = prune_to
    svs_params.use_search_history = full_search_history
    svs_params.search_window_size = search_window_size
    svs_params.epsilon = epsilon
    svs_params.num_threads = num_threads

    return SVSIndex(svs_params)

def compute_k_euclidean(dataset, query, k):
    dists = [(spatial.distance.euclidean(query, vec), key) for key, vec in dataset]
    dists = sorted(dists)
    return dists[:k]

def compute_k_cosine(dataset, query, k):
    dists = [(spatial.distance.cosine(query, vec), key) for key, vec in dataset]
    dists = sorted(dists)
    return dists[:k]

def compute_range_euclidean(dataset, query, radius):
    dists = [(spatial.distance.sqeuclidean(query, vec), key) for key, vec in dataset]
    return sorted([(dist, key) for dist, key in dists if dist <= radius])

def extract_labels(dists):
    return [key for _, key in dists]

def extract_dists(dists):
    return [dist for dist, _ in dists]

def count_correctness(actual_labels, desired_labels):
    correct = 0
    for label in actual_labels:
        if label in desired_labels:
            correct += 1
    return correct


# compare results with the original version of hnswlib - do not use elements deletion.
def test_sanity_svs_index_L2(test_logger):
    dim = 16
    num_elements = 10000
    k = 10

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_L2)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))
    redis_labels, redis_distances = index.knn_query(query_data, k)
    desired = compute_k_euclidean(vectors, query_data[0], k)
    desired_labels = [key for _, key in desired]
    count = count_correctness(desired_labels, redis_labels[0])
    recall = float(count) / k
    test_logger.info(f"recall is: {recall}")
    assert(recall > 0.9)


def test_sanity_svs_index_cosine(test_logger):
    dim = 16
    num_elements = 10000
    k = 10

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_Cosine, alpha=0.9)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))
    redis_labels, redis_distances = index.knn_query(query_data, k)
    desired = compute_k_cosine(vectors, query_data[0], k)
    desired_labels = [key for _, key in desired]
    count = count_correctness(desired_labels, redis_labels[0])
    recall = float(count) / k
    test_logger.info(f"recall is: {recall}")
    assert(recall > 0.9)


# Validate correctness of delete implementation comparing the brute force search. We test the search recall which is not
# deterministic, but should be above a certain threshold. Note that recall is highly impacted by changing
# index parameters.
def test_recall_for_svs_index_with_deletion(test_logger):
    dim = 16
    num_elements = 10000

    num_queries = 10
    k = 10

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_L2, search_window_size=50)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    # delete half of the data
    for i in range(0, len(data), 2):
        index.delete_vector(i)
    vectors = [vectors[i] for i in range(1, len(data), 2)]

    query_data = np.float32(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        redis_labels, redis_distances = index.knn_query(target_vector, k)
        dists = compute_k_euclidean(vectors, target_vector, k)
        keys = extract_labels(dists)
        correct += count_correctness(redis_labels[0], keys)

    # Measure recall
    recall = float(correct) / (k * num_queries)
    test_logger.info(f"recall is: {recall}")
    assert (recall > 0.9)


def test_batch_iterator(test_logger):
    dim = 100
    num_elements = 10000
    num_queries = 10
    windowSize = 128

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_L2,
                             window_size=windowSize, search_window_size=windowSize)

    # Add 100k random vectors to the index
    rng = np.random.default_rng(seed=47)
    data = np.float32(rng.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        index.add_vector(vector, i)
        vectors.append((i, vector))

    # Create a random query vector and create a batch iterator
    query_data = np.float32(rng.random((1, dim)))
    batch_iterator = index.create_batch_iterator(query_data)
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
    query_params.svsRuntimeParams.windowSize = 5
    batch_iterator_new = index.create_batch_iterator(query_data, query_params)
    labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
    # Verify that accuracy is worse with the new lower window size.
    assert (sum(distances_first_batch[0]) < sum(distances_first_batch_new[0]))

    query_params.svsRuntimeParams.windowSize = windowSize  # Restore previous window size.
    batch_iterator_new = index.create_batch_iterator(query_data, query_params)
    labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
    # Verify that results are now the same.
    assert_allclose(distances_first_batch_new[0], distances_first_batch[0])
    assert_equal(labels_first_batch_new[0], labels_first_batch[0])

    # Reset
    batch_iterator.reset()

    # Run in batches of 100 until we reach 1000 results and measure recall
    batch_size = 100
    total_res = 1000
    total_recall = 0
    query_data = np.float32(rng.random((num_queries, dim)))
    for target_vector in query_data:
        batch_iterator = index.create_batch_iterator(target_vector)
        iterations = 0
        dists = compute_k_euclidean(vectors, target_vector, total_res)
        accumulated_labels = []
        returned_results_num = 0
        while batch_iterator.has_next() and returned_results_num < total_res:
            iterations += 1
            labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
            accumulated_labels.extend(labels[0])
            returned_results_num = len(accumulated_labels)

        keys = extract_labels(dists)
        correct = count_correctness(accumulated_labels, keys)

        assert iterations == np.ceil(total_res / batch_size)
        recall = float(correct) / total_res
        assert recall >= 0.89
        total_recall += recall
    test_logger.info(f'Avg recall for {total_res} results in index of size {num_elements} with dim={dim} is: {total_recall / num_queries}')

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


def test_topk_query(test_logger):
    dim = 128
    num_elements = 100000

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_L2)

    np.random.seed(47)
    start = time.time()
    data = np.float32(np.random.random((num_elements, dim)))
    test_logger.info(f'Sample data generated in {time.time() - start} seconds')
    vectors = []
    start = time.time()
    for i, vector in enumerate(data):
        vectors.append((i, vector))

    index.add_vector_parallel(data, np.array(range(num_elements)))
    test_logger.info(f'Index built in {time.time() - start} seconds')

    query_data = np.float32(np.random.random((1, dim)))

    k = 128
    recalls = {}

    for window_size in [128, 256, 512]:
        query_params = VecSimQueryParams()
        query_params.svsRuntimeParams.windowSize = window_size
        query_params.svsRuntimeParams.searchHistory = VecSimOption_AUTO
        start = time.time()
        redis_labels, redis_distances = index.knn_query(query_data, k, query_param=query_params)
        end = time.time()
        assert len(redis_labels[0]) == k

        actual_results = compute_k_euclidean(vectors, query_data.flat, k)
        assert len(actual_results) == k

        keys = extract_labels(actual_results)
        correct = count_correctness(redis_labels[0], keys)

        test_logger.info(
            f'lookup time for {num_elements} vectors with dim={dim} took {end - start} seconds with window_size={window_size},'
            f' got {correct} correct results, which are {correct / k} of the entire results in the range.')

        recalls[window_size] = correct / k

    # Expect higher recalls for higher epsilon values.
    assert recalls[128] <= recalls[256] <= recalls[512]

    # Expect zero results for radius==0
    redis_labels, redis_distances = index.knn_query(query_data, 0)
    assert len(redis_labels[0]) == 0


def test_range_query(test_logger):
    dim = 100
    num_elements = 100000

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_L2)

    np.random.seed(47)
    start = time.time()
    data = np.float32(np.random.random((num_elements, dim)))
    test_logger.info(f'Sample data generated in {time.time() - start} seconds')
    vectors = []
    start = time.time()
    for i, vector in enumerate(data):
        vectors.append((i, vector))

    index.add_vector_parallel(data, np.array(range(num_elements)))
    test_logger.info(f'Index built in {time.time() - start} seconds')

    query_data = np.float32(np.random.random((1, dim)))

    radius = 13.0
    recalls = {}

    for epsilon_rt in [0.001, 0.01, 0.1]:
        query_params = VecSimQueryParams()
        query_params.svsRuntimeParams.epsilon = epsilon_rt
        start = time.time()
        redis_labels, redis_distances = index.range_query(query_data, radius=radius, query_param=query_params)
        end = time.time()
        res_num = len(redis_labels[0])

        actual_results = compute_range_euclidean(vectors, query_data.flat, radius)

        test_logger.info(
            f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds with window_size={epsilon_rt},'
            f' got {res_num} results, which are {res_num / len(actual_results)} of the entire results in the range.')

        # Compare the number of vectors that are actually within the range to the returned results.
        assert np.all(np.isin(redis_labels, np.array([label for _, label in actual_results])))

        assert max(redis_distances[0]) <= radius
        recalls[epsilon_rt] = res_num / len(actual_results)

    # Expect higher recalls for higher epsilon values.
    assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

    # Expect zero results for radius==0
    redis_labels, redis_distances = index.range_query(query_data, radius=0)
    assert len(redis_labels[0]) == 0

def test_recall_for_svs_multi_value(test_logger):
    dim = 16
    num_labels = 1000
    num_per_label = 4
    num_queries = 10
    k = 10

    num_elements = num_labels * num_per_label

    svs_index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_Cosine, alpha=0.9,
                                 search_window_size=50, is_multi=True)

    np.random.seed(47)
    data = np.float32(np.random.random((num_labels, dim)))
    vectors = []
    for i, vector in enumerate(data):
        for _ in range(num_per_label):
            svs_index.add_vector(vector, i)
            vectors.append((i, vector))

    query_data = np.float32(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        svs_labels, svs_distances = svs_index.knn_query(target_vector, k)
        assert (len(svs_labels[0]) == len(np.unique(svs_labels[0])))

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

        for label in svs_labels[0]:
            for correct_label in keys:
                if label == correct_label:
                    correct += 1
                    break

    # Measure recall
    recall = float(correct) / (k * num_queries)
    test_logger.info(f"recall is: {recall}")
    assert (recall > 0.9)


def test_multi_range_query(test_logger):
    dim = 100
    num_labels = 10000
    per_label = 5
    num_elements = num_labels * per_label

    index = create_svs_index(dim, num_elements, VecSimType_FLOAT32, VecSimMetric_L2, is_multi=True)

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
        query_params.svsRuntimeParams.epsilon = epsilon_rt
        start = time.time()
        svs_labels, svs_distances = index.range_query(query_data, radius=radius, query_param=query_params)
        end = time.time()
        res_num = len(svs_labels[0])

        test_logger.info(
            f'lookup time for ({num_labels} X {per_label}) vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
            f' got {res_num} results, which are {res_num / len(keys)} of the entire results in the range.')

        # Compare the number of vectors that are actually within the range to the returned results.
        assert np.all(np.isin(svs_labels, np.array(keys)))

        # Asserts that all the results are unique
        assert len(svs_labels[0]) == len(np.unique(svs_labels[0]))

        assert max(svs_distances[0]) <= radius
        recalls[epsilon_rt] = res_num / len(keys)

    # Expect higher recalls for higher epsilon values.
    assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

    # Expect zero results for radius==0
    svs_labels, svs_distances = index.range_query(query_data, radius=0)
    assert len(svs_labels[0]) == 0
