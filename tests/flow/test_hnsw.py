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
import VecSim
from common import *
import hnswlib


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
def test_recall_for_hnswlib_index_with_deletion(test_logger):
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
    test_logger.info(f"recall is: {recall}")
    assert (recall > 0.9)


def test_batch_iterator(test_logger):
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
    assert (sum(distances_first_batch[0]) < sum(distances_first_batch_new[0]))

    query_params.hnswRuntimeParams.efRuntime = efRuntime  # Restore previous ef_runtime.
    batch_iterator_new = hnsw_index.create_batch_iterator(query_data, query_params)
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
    test_logger.info(f'Avg recall for {total_res} results in index of size {num_elements} with dim={dim} is: {total_recall / num_queries}')

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
    test_logger.info(f"Overall results returned: {len(accumulated_labels)} in {iterations} iterations")


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
    logger.info(f"recall is: {recall}")

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
    logger.info(f"recall after is: {recall_after}")
    assert recall == recall_after


def test_range_query(test_logger):
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

        test_logger.info(
            f'lookup time for {num_elements} vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
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


def test_recall_for_hnsw_multi_value(test_logger):
    dim = 16
    num_labels = 1000
    num_per_label = 16
    M = 16
    efConstruction = 100
    num_queries = 10
    k = 10
    efRuntime = 0

    num_elements = num_labels * num_per_label

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_Cosine, VecSimType_FLOAT32, efConstruction, M,
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
    test_logger.info(f"recall is: {recall}")
    assert (recall > 0.9)


def test_multi_range_query(test_logger):
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

        test_logger.info(
            f'lookup time for ({num_labels} X {per_label}) vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
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

class TestBfloat16():
    dim = 50
    num_elements = 10_000
    M = 32
    efConstruction = 200
    efRuntime = 50
    data_type = VecSimType_BFLOAT16

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, data_type, efConstruction, M, efRuntime)
    hnsw_index.set_ef(efRuntime)

    rng = np.random.default_rng(seed=42)
    data = vec_to_bfloat16(rng.random((num_elements, dim)))

    vectors = []
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    #### Create queries
    num_queries = 10
    query_data = vec_to_bfloat16(rng.random((num_queries, dim)))

    def test_serialization(self, test_logger):
        hnsw_index = self.hnsw_index
        k = 10

        correct = 0
        correct_labels = []  # cache these
        for target_vector in self.query_data:
            hnswlib_labels, _ = hnsw_index.knn_query(target_vector, 10)

            # sort distances of every vector from the target vector and get actual k nearest vectors
            dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in self.vectors]
            dists = sorted(dists)
            keys = [key for _, key in dists[:k]]
            correct_labels.append(keys)

            for label in hnswlib_labels[0]:
                for correct_label in keys:
                    if label == correct_label:
                        correct += 1
                        break
        # Measure recall
        recall = float(correct) / (k * self.num_queries)
        test_logger.info(f"recall is: {recall}")

        # Persist, delete and restore index.
        file_name = os.getcwd() + "/dump"
        hnsw_index.save_index(file_name)

        new_hnsw_index = HNSWIndex(file_name)
        os.remove(file_name)
        assert new_hnsw_index.index_size() == self.num_elements
        assert new_hnsw_index.index_type() == VecSimType_BFLOAT16

        # Check recall
        correct_after = 0
        for i, target_vector in enumerate(self.query_data):
            hnswlib_labels, _ = new_hnsw_index.knn_query(target_vector, 10)
            correct_labels_cur = correct_labels[i]
            for label in hnswlib_labels[0]:
                for correct_label in correct_labels_cur:
                    if label == correct_label:
                        correct_after += 1
                        break

        # Compare recall after reloading the index
        recall_after = float(correct_after) / (k * self.num_queries)
        test_logger.info(f"recall after is: {recall_after}")
        assert recall == recall_after

    def test_bfloat16_L2(self, test_logger):
        hnsw_index = self.hnsw_index
        k = 10

        correct = 0
        for target_vector in self.query_data:
            hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

            results, keys = get_ground_truth_results(spatial.distance.sqeuclidean, target_vector, self.vectors, k)
            for i, label in enumerate(hnswlib_labels[0]):
                for j, correct_label in enumerate(keys):
                    if label == correct_label:
                        correct += 1
                        assert math.isclose(hnswlib_distances[0][i], results[j]["dist"], rel_tol=1e-5)
                        break

        # Measure recall
        recall = float(correct) / (k * self.num_queries)
        test_logger.info(f"recall is: {recall}")
        assert (recall > 0.9)

    def test_batch_iterator(self):
        hnsw_index = self.hnsw_index

        efRuntime = 180
        hnsw_index.set_ef(efRuntime)

        batch_iterator = hnsw_index.create_batch_iterator(self.query_data)
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
        batch_iterator_new = hnsw_index.create_batch_iterator(self.query_data, query_params)
        labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
        # Verify that accuracy is worse with the new lower ef_runtime.
        assert (sum(distances_first_batch[0]) < sum(distances_first_batch_new[0]))

    def test_range_query(self, test_logger):
        index = self.hnsw_index

        radius = 7.0
        recalls = {}

        for epsilon_rt in [0.001, 0.01, 0.1]:
            query_params = VecSimQueryParams()
            query_params.hnswRuntimeParams.epsilon = epsilon_rt
            start = time.time()
            hnsw_labels, hnsw_distances = index.range_query(self.query_data[0], radius=radius, query_param=query_params)
            end = time.time()
            res_num = len(hnsw_labels[0])

            dists = sorted([(key, spatial.distance.sqeuclidean(self.query_data[0], vec)) for key, vec in self.vectors])
            actual_results = [(key, dist) for key, dist in dists if dist <= radius]

            test_logger.info(
                f'lookup time for {self.num_elements} vectors with dim={self.dim} took {end - start} seconds with epsilon={epsilon_rt},'
                f' got {res_num} results, which are {res_num / len(actual_results)} of the entire results in the range.')

            # Compare the number of vectors that are actually within the range to the returned results.
            assert np.all(np.isin(hnsw_labels, np.array([label for label, _ in actual_results])))

            assert max(hnsw_distances[0]) <= radius
            recalls[epsilon_rt] = res_num / len(actual_results)

        # Expect higher recalls for higher epsilon values.
        assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

        # Expect zero results for radius==0
        hnsw_labels, hnsw_distances = index.range_query(self.query_data[0], radius=0)
        assert len(hnsw_labels[0]) == 0

def test_hnsw_bfloat16_multi_value(test_logger):
    num_labels = 1_000
    num_per_label = 5
    num_elements = num_labels * num_per_label

    dim = 128
    M = 32
    efConstruction = 100
    num_queries = 10

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_BFLOAT16, efConstruction, M,
                                   is_multi=True)
    k = 10
    hnsw_index.set_ef(50)

    data = vec_to_bfloat16(np.random.random((num_labels, dim)))
    vectors = []
    for i, vector in enumerate(data):
        for _ in range(num_per_label):
            hnsw_index.add_vector(vector, i)
            vectors.append((i, vector))

    query_data = vec_to_bfloat16(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)
        assert (len(hnswlib_labels[0]) == len(np.unique(hnswlib_labels[0])))

        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = {}
        for key, vec in vectors:
            # Setting or updating the score for each label.
            # If it's the first time we calculate a score for a label dists.get(key, dist)
            # will return dist so we will choose the actual score the first time.
            dist = spatial.distance.sqeuclidean(target_vector, vec)
            dists[key] = min(dist, dists.get(key, dist))

        dists = list(dists.items())
        dists = sorted(dists, key=lambda pair: pair[1])[:k]
        keys = [key for key, _ in dists]

        for i, label in enumerate(hnswlib_labels[0]):
            for j, correct_label in enumerate(keys):
                if label == correct_label:
                    correct += 1
                    assert math.isclose(hnswlib_distances[0][i], dists[j][1], rel_tol=1e-5)
                    break

    # Measure recall
    recall = float(correct) / (k * num_queries)
    test_logger.info(f"recall is: {recall}")
    assert (recall > 0.9)

class TestFloat16():
    dim = 50
    num_elements = 10_000
    M = 32
    efConstruction = 200
    efRuntime = 50
    data_type = VecSimType_FLOAT16

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, data_type, efConstruction, M, efRuntime)
    hnsw_index.set_ef(efRuntime)

    rng = np.random.default_rng(seed=42)
    data = vec_to_float16(rng.random((num_elements, dim)))

    vectors = []
    for i, vector in enumerate(data):
        hnsw_index.add_vector(vector, i)
        vectors.append((i, vector))

    #### Create queries
    num_queries = 10
    query_data = vec_to_float16(rng.random((num_queries, dim)))

    def test_serialization(self, test_logger):
        hnsw_index = self.hnsw_index
        k = 10

        correct = 0
        correct_labels = []  # cache these
        for target_vector in self.query_data:
            hnswlib_labels, _ = hnsw_index.knn_query(target_vector, 10)

            # sort distances of every vector from the target vector and get actual k nearest vectors
            dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in self.vectors]
            dists = sorted(dists)
            keys = [key for _, key in dists[:k]]
            correct_labels.append(keys)

            for label in hnswlib_labels[0]:
                for correct_label in keys:
                    if label == correct_label:
                        correct += 1
                        break
        # Measure recall
        recall = float(correct) / (k * self.num_queries)
        test_logger.info(f"recall is: {recall}")

        # Persist, delete and restore index.
        file_name = os.getcwd() + "/dump"
        hnsw_index.save_index(file_name)

        new_hnsw_index = HNSWIndex(file_name)
        os.remove(file_name)
        assert new_hnsw_index.index_size() == self.num_elements
        assert new_hnsw_index.index_type() == VecSimType_FLOAT16

        # Check recall
        correct_after = 0
        for i, target_vector in enumerate(self.query_data):
            hnswlib_labels, _ = new_hnsw_index.knn_query(target_vector, 10)
            correct_labels_cur = correct_labels[i]
            for label in hnswlib_labels[0]:
                for correct_label in correct_labels_cur:
                    if label == correct_label:
                        correct_after += 1
                        break

        # Compare recall after reloading the index
        recall_after = float(correct_after) / (k * self.num_queries)
        test_logger.info(f"recall after is: {recall_after}")
        assert recall == recall_after

    def test_float16_L2(self, test_logger):
        hnsw_index = self.hnsw_index
        k = 10

        correct = 0
        for target_vector in self.query_data:
            hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

            results, keys = get_ground_truth_results(spatial.distance.sqeuclidean, target_vector, self.vectors, k)
            for i, label in enumerate(hnswlib_labels[0]):
                for j, correct_label in enumerate(keys):
                    if label == correct_label:
                        correct += 1
                        assert math.isclose(np.float16(hnswlib_distances[0][i]), (results[j]["dist"]), rel_tol=1e-2), f"label: {label}"
                        break

        # Measure recall
        recall = float(correct) / (k * self.num_queries)
        test_logger.info(f"recall is: {recall}")
        assert (recall > 0.9)

    def test_batch_iterator(self):
        hnsw_index = self.hnsw_index

        efRuntime = 180
        hnsw_index.set_ef(efRuntime)

        batch_iterator = hnsw_index.create_batch_iterator(self.query_data)
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
        batch_iterator_new = hnsw_index.create_batch_iterator(self.query_data, query_params)
        labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
        # Verify that accuracy is worse with the new lower ef_runtime.
        assert (sum(distances_first_batch[0]) < sum(distances_first_batch_new[0]))

    def test_range_query(self, test_logger):
        index = self.hnsw_index

        radius = 7.0
        recalls = {}

        for epsilon_rt in [0.001, 0.01, 0.1]:
            query_params = VecSimQueryParams()
            query_params.hnswRuntimeParams.epsilon = epsilon_rt
            start = time.time()
            hnsw_labels, hnsw_distances = index.range_query(self.query_data[0], radius=radius, query_param=query_params)
            end = time.time()
            res_num = len(hnsw_labels[0])

            dists = sorted([(key, spatial.distance.sqeuclidean(self.query_data[0], vec)) for key, vec in self.vectors])
            actual_results = [(key, dist) for key, dist in dists if dist <= radius]

            test_logger.info(
                f'lookup time for {self.num_elements} vectors with dim={self.dim} took {end - start} seconds with epsilon={epsilon_rt},'
                f' got {res_num} results, which are {res_num / len(actual_results)} of the entire results in the range.')

            # Compare the number of vectors that are actually within the range to the returned results.
            assert np.all(np.isin(hnsw_labels, np.array([label for label, _ in actual_results])))

            assert max(hnsw_distances[0]) <= radius
            recalls[epsilon_rt] = res_num / len(actual_results)

        # Expect higher recalls for higher epsilon values.
        assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

        # Expect zero results for radius==0
        hnsw_labels, hnsw_distances = index.range_query(self.query_data[0], radius=0)
        assert len(hnsw_labels[0]) == 0

def test_hnsw_float16_multi_value(test_logger):
    num_labels = 1_000
    num_per_label = 5
    num_elements = num_labels * num_per_label

    dim = 128
    M = 32
    efConstruction = 100
    num_queries = 10

    hnsw_index = create_hnsw_index(dim, num_elements, VecSimMetric_L2, VecSimType_FLOAT16, efConstruction, M,
                                   is_multi=True)
    k = 10
    hnsw_index.set_ef(50)

    data = vec_to_float16(np.random.random((num_labels, dim)))
    vectors = []
    for i, vector in enumerate(data):
        for _ in range(num_per_label):
            hnsw_index.add_vector(vector, i)
            vectors.append((i, vector))

    query_data = vec_to_float16(np.random.random((num_queries, dim)))
    correct = 0
    for target_vector in query_data:
        hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)
        assert (len(hnswlib_labels[0]) == len(np.unique(hnswlib_labels[0])))

        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = {}
        for key, vec in vectors:
            # Setting or updating the score for each label.
            # If it's the first time we calculate a score for a label dists.get(key, dist)
            # will return dist so we will choose the actual score the first time.
            dist = spatial.distance.sqeuclidean(target_vector, vec)
            dists[key] = min(dist, dists.get(key, dist))

        dists = list(dists.items())
        dists = sorted(dists, key=lambda pair: pair[1])[:k]
        keys = [key for key, _ in dists]

        for i, label in enumerate(hnswlib_labels[0]):
            for j, correct_label in enumerate(keys):
                if label == correct_label:
                    correct += 1
                    assert math.isclose(np.float16(hnswlib_distances[0][i]), dists[j][1], rel_tol=1e-2)
                    break

    # Measure recall
    recall = float(correct) / (k * num_queries)
    test_logger.info(f"recall is: {recall}")
    assert (recall > 0.9)

'''
A Class to run common tests for HNSW index

The following tests will *automatically* run if the class is inherited:
* test_serialization - single L2 index
* test_L2 - single L2 index
* test_batch_iterator - single L2 index

The following tests should be *explicitly* called from a method prefixed with test_*
# range_query(dist_func) - single cosine index

@param create_data_func is a function expects num_elements, dim, [and optional np.random.Generator] as input and
returns a (num_elements, dim) numpy array of vectors
uses multi L2 index
# multi_value(create_data_func, num_per_label) -
'''
class GeneralTest():
    dim = 50
    num_elements = 10_000
    num_queries = 10
    M = 32
    efConstruction = 200
    efRuntime = 50
    data_type = None

    rng = np.random.default_rng(seed=42)
    data = None
    query_data = None

    # single HNSW index with L2 metric
    cache_hnsw_index_L2_single = None
    cached_label_to_vec_list = None

    @classmethod
    def create_index(cls, metric = VecSimMetric_L2, is_multi=False):
        assert cls.data_type is not None
        hnsw_index = create_hnsw_index(cls.dim, 0, metric, cls.data_type, cls.efConstruction, cls.M, cls.efRuntime, is_multi=is_multi)
        return hnsw_index

    @classmethod
    def create_add_vectors(cls, hnsw_index):
        assert cls.data is not None
        return create_add_vectors(hnsw_index, cls.data)

    @classmethod
    def get_cached_single_L2_index(cls):
        if cls.cache_hnsw_index_L2_single is None:
            cls.cache_hnsw_index_L2_single = cls.create_index()
            cls.cached_label_to_vec_list = cls.create_add_vectors(cls.cache_hnsw_index_L2_single)
        return cls.cache_hnsw_index_L2_single, cls.cached_label_to_vec_list

    @staticmethod
    def compute_correct(res_labels, res_dist, gt_labels, gt_dist_label_list):
        correct = 0
        for i, label in enumerate(res_labels):
            for j, correct_label in enumerate(gt_labels):
                if label == correct_label:
                    correct += 1
                    assert math.isclose(res_dist[i], gt_dist_label_list[j]["dist"], rel_tol=1e-5)
                    break

        return correct

    @classmethod
    def knn(cls, hnsw_index, label_vec_list, dist_func, test_logger):
        k = 10

        correct = 0
        for target_vector in cls.query_data:
            hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, k)
            results, keys = get_ground_truth_results(dist_func, target_vector, label_vec_list, k)

            correct += cls.compute_correct(hnswlib_labels[0], hnswlib_distances[0], keys, results)

        # Measure recall
        recall = recall = float(correct) / (k * cls.num_queries)
        test_logger.info(f"recall is: {recall}")
        assert (recall > 0.9)

    def test_serialization(self, test_logger):
        assert self.data_type is not None
        hnsw_index, label_to_vec_list = self.get_cached_single_L2_index()
        k = 10

        correct = 0
        correct_labels = []  # cache these
        for target_vector in self.query_data:
            hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, k)
            results, keys = get_ground_truth_results(spatial.distance.sqeuclidean, target_vector, label_to_vec_list, k)

            correct_labels.append(keys)
            correct += self.compute_correct(hnswlib_labels[0], hnswlib_distances[0], keys, results)

        # Measure recall
        recall = float(correct) / (k * self.num_queries)
        test_logger.info(f"recall is: {recall}")

        # Persist, delete and restore index.
        file_name = os.getcwd() + "/dump"
        hnsw_index.save_index(file_name)

        new_hnsw_index = HNSWIndex(file_name)
        os.remove(file_name)
        assert new_hnsw_index.index_size() == self.num_elements
        assert new_hnsw_index.index_type() == self.data_type
        assert new_hnsw_index.check_integrity()

        # Check recall
        correct_after = 0
        for i, target_vector in enumerate(self.query_data):
            hnswlib_labels, _ = new_hnsw_index.knn_query(target_vector, k)
            correct_labels_cur = correct_labels[i]
            for label in hnswlib_labels[0]:
                for correct_label in correct_labels_cur:
                    if label == correct_label:
                        correct_after += 1
                        break

        # Compare recall after reloading the index
        recall_after = float(correct_after) / (k * self.num_queries)
        test_logger.info(f"recall after is: {recall_after}")
        assert recall == recall_after

    def test_L2(self, test_logger):
        hnsw_index, label_to_vec_list = self.get_cached_single_L2_index()
        self.knn(hnsw_index, label_to_vec_list, spatial.distance.sqeuclidean, test_logger)

    def test_batch_iterator(self):
        hnsw_index, _ = self.get_cached_single_L2_index()

        batch_size = 10

        efRuntime = 180
        hnsw_index.set_ef(efRuntime)

        batch_iterator = hnsw_index.create_batch_iterator(self.query_data)
        labels_first_batch, distances_first_batch = batch_iterator.get_next_results(batch_size, BY_ID)
        for i, _ in enumerate(labels_first_batch[0][:-1]):
            # Assert sorting by id
            assert (labels_first_batch[0][i] < labels_first_batch[0][i + 1])

        _, distances_second_batch = batch_iterator.get_next_results(batch_size, BY_SCORE)
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
        batch_iterator_new = hnsw_index.create_batch_iterator(self.query_data, query_params)
        _, distances_first_batch_new = batch_iterator_new.get_next_results(batch_size, BY_ID)
        # Verify that accuracy is worse with the new lower ef_runtime.
        assert (sum(distances_first_batch[0]) < sum(distances_first_batch_new[0]))

        # reset efRuntime
        hnsw_index.set_ef(self.efRuntime)

    ##### Should be explicitly called #####

    def range_query(self, dist_func, test_logger):
        hnsw_index = self.create_index(VecSimMetric_Cosine)
        label_to_vec_list = self.create_add_vectors(hnsw_index)
        radius = hnsw_index.knn_query(self.query_data[0], k=100)[1][0][-1] # get the distance of the 100th closest vector as the radius
        recalls = {}

        for epsilon_rt in [0.001, 0.01, 0.1]:
            query_params = VecSimQueryParams()
            query_params.hnswRuntimeParams.epsilon = epsilon_rt
            start = time.time()
            hnsw_labels, hnsw_distances = hnsw_index.range_query(self.query_data[0], radius=radius, query_param=query_params)
            end = time.time()
            res_num = len(hnsw_labels[0])

            dists = sorted([(key, dist_func(self.query_data[0], vec)) for key, vec in label_to_vec_list])
            actual_results = [(key, dist) for key, dist in dists if dist <= radius]

            test_logger.info(
                f'lookup time for {self.num_elements} vectors with dim={self.dim} took {end - start} seconds with epsilon={epsilon_rt},'
                f' got {res_num} results, which are {res_num / len(actual_results)} of the entire results in the range.')

            # Compare the number of vectors that are actually within the range to the returned results.
            assert np.all(np.isin(hnsw_labels, np.array([label for label, _ in actual_results])))

            assert max(hnsw_distances[0]) <= radius
            recall = res_num / len(actual_results)
            assert recall > 0.9
            recalls[epsilon_rt] = res_num / len(actual_results)

        # Expect higher recalls for higher epsilon values.
        assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

        # Expect zero results for radius==0
        hnsw_labels, hnsw_distances = hnsw_index.range_query(self.query_data[0], radius=0)
        assert len(hnsw_labels[0]) == 0

    def multi_value(self, create_data_func, test_logger, num_per_label = 5):
        num_per_label = 5
        num_labels = self.num_elements // num_per_label
        k = 10

        data = create_data_func((num_labels, num_per_label, self.dim), self.rng)

        hnsw_index = self.create_index(is_multi=True)

        vectors = []
        for i, cur_vectors in enumerate(data):
            for vector in cur_vectors:
                hnsw_index.add_vector(vector, i)
                vectors.append((i, vector))

        correct = 0
        for target_vector in self.query_data:
            hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, k)
            assert (len(hnswlib_labels[0]) == len(np.unique(hnswlib_labels[0])))

            # sort distances of every vector from the target vector and get actual k nearest vectors
            dists = {}
            for key, vec in vectors:
                # Setting or updating the score for each label.
                # If it's the first time we calculate a score for a label dists.get(key, dist)
                # will return dist so we will choose the actual score the first time.
                dist = spatial.distance.sqeuclidean(target_vector, vec)
                dists[key] = min(dist, dists.get(key, dist))

            dists = list(dists.items())
            dists = sorted(dists, key=lambda pair: pair[1])[:k]
            keys = [key for key, _ in dists]

            for i, label in enumerate(hnswlib_labels[0]):
                for j, correct_label in enumerate(keys):
                    if label == correct_label:
                        correct += 1
                        assert math.isclose(hnswlib_distances[0][i], dists[j][1], rel_tol=1e-5)
                        break

        # Measure recall
        recall = float(correct) / (k * self.num_queries)
        test_logger.info(f"recall is: {recall}")
        assert (recall > 0.9)

class TestINT8(GeneralTest):

    data_type = VecSimType_INT8

    #### Create vectors
    data = create_int8_vectors((GeneralTest.num_elements, GeneralTest.dim), GeneralTest.rng)

    #### Create queries
    query_data = create_int8_vectors((GeneralTest.num_queries, GeneralTest.dim), GeneralTest.rng)

    def test_Cosine(self, test_logger):
        hnsw_index = self.create_index(VecSimMetric_Cosine)
        label_to_vec_list = self.create_add_vectors(hnsw_index)

        self.knn(hnsw_index, label_to_vec_list, fp32_expand_and_calc_cosine_dist, test_logger)

    def test_range_query(self, test_logger):
        self.range_query(fp32_expand_and_calc_cosine_dist, test_logger)

    def test_multi_value(self, test_logger):
        self.multi_value(create_int8_vectors, test_logger)

class TestUINT8(GeneralTest):

    data_type = VecSimType_UINT8

    #### Create vectors
    data = create_uint8_vectors((GeneralTest.num_elements, GeneralTest.dim), GeneralTest.rng)

    #### Create queries
    query_data = create_uint8_vectors((GeneralTest.num_queries, GeneralTest.dim), GeneralTest.rng)

    def test_Cosine(self, test_logger):
        hnsw_index = self.create_index(VecSimMetric_Cosine)
        label_to_vec_list = self.create_add_vectors(hnsw_index)

        self.knn(hnsw_index, label_to_vec_list, fp32_expand_and_calc_cosine_dist, test_logger)

    def test_range_query(self, test_logger):
        self.range_query(fp32_expand_and_calc_cosine_dist, test_logger)

    def test_multi_value(self, test_logger):
        self.multi_value(create_uint8_vectors, test_logger)
