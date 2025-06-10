# Copyright (c) 2006-Present, Redis Ltd.
# All rights reserved.
#
# Licensed under your choice of the Redis Source Available License 2.0
# (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
# GNU Affero General Public License v3 (AGPLv3).

import logging
from common import *

class Data:
    def __init__(self, data_type, metric, dist_func, np_fuc, dim=16, num_labels=10, num_per_label=1):
        bfparams = BFParams()

        bfparams.blockSize = num_labels
        bfparams.dim = dim
        bfparams.type = data_type
        bfparams.metric = metric
        bfparams.multi = num_per_label > 1


        self.index = BFIndex(bfparams)

        self.metric = metric
        self.type = data_type
        self.dist_func = dist_func

        num_elements = num_labels * num_per_label

        np.random.seed(47)
        self.data = np_fuc(np.random.random((num_elements, dim)))
        self.query = np_fuc(np.random.random((1, dim)))
        self.vectors = []
        for i, vector in enumerate(self.data):
            self.index.add_vector(vector, i % num_labels)
            self.vectors.append((i % num_labels, vector))

    def measure_dists(self, k):
        dists = [(self.dist_func(self.query.flat, vec), key) for key, vec in self.vectors]
        dists = sorted(dists)[:k]
        keys = [key for _, key in dists]
        dists = [dist for dist, _ in dists]
        return (keys, dists)

def test_sanity_bf(test_logger):
    test_datas = []

    dist_funcs = [(VecSimMetric_Cosine, spatial.distance.cosine), (VecSimMetric_L2, spatial.distance.sqeuclidean)]
    types = [(VecSimType_FLOAT32, np.float32), (VecSimType_FLOAT64, np.float64)]
    for type_name, np_type in types:
        for dist_name, dist_func in dist_funcs:
            test_datas.append(Data(type_name, dist_name, dist_func, np_type))

    k = 10
    for test_data in test_datas:

        keys, dists = test_data.measure_dists(k)
        bf_labels, bf_distances = test_data.index.knn_query(test_data.query, k=k)
        assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
        assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)
        test_logger.info(f"sanity test for {test_data.metric} and {test_data.type} pass")

def test_bf_cosine():
    dim = 128
    num_elements = 1000000
    k=10

    bfparams = BFParams()
    bfparams.blockSize = num_elements
    bfparams.dim = dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_Cosine
    bfindex = BFIndex(bfparams)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))

    dists = [(spatial.distance.cosine(query_data.flat, vec), key) for key, vec in vectors]
    dists = sorted(dists)[:k]
    keys = [key for _, key in dists[:k]]
    dists = [dist for dist, _ in dists[:k]]
    start = time.time()
    bf_labels, bf_distances = bfindex.knn_query(query_data, k=10)
    end = time.time()
    logger.info(f'lookup time for {num_elements} vectors with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)


def test_bf_l2(test_logger):
    dim = 128
    num_elements = 1000000
    k=10

    bfparams = BFParams()
    bfparams.blockSize = num_elements
    bfparams.dim = dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_L2
    bfindex = BFIndex(bfparams)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))

    dists = [(spatial.distance.euclidean(query_data.flat, vec), key) for key, vec in vectors]
    dists = sorted(dists)[:k]
    keys = [key for _, key in dists[:k]]
    start = time.time()
    bf_labels, bf_distances = bfindex.knn_query(query_data, k=10)
    end = time.time()
    test_logger.info(f'lookup time for {num_elements} vectors with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)


def test_batch_iterator(test_logger):
    dim = 128
    num_elements = 1000000

    # Create a brute force index for vectors of 128 floats. Use 'Cosine' as the distance metric
    bfparams = BFParams()
    bfparams.blockSize = num_elements
    bfparams.dim = dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_L2
    bf_index = BFIndex(bfparams)

    # Add 1M random vectors to the index
    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    for i, vector in enumerate(data):
        bf_index.add_vector(vector, i)

    # Create a random query vector and create a batch iterator
    query_data = np.float32(np.random.random((1, dim)))
    batch_iterator = bf_index.create_batch_iterator(query_data)
    labels_first_batch, distances_first_batch = batch_iterator.get_next_results(10, BY_ID)
    for i, _ in enumerate(labels_first_batch[0][:-1]):
        # assert sorting by id
        assert(labels_first_batch[0][i] < labels_first_batch[0][i+1])

    labels_second_batch, distances_second_batch = batch_iterator.get_next_results(10, BY_SCORE)
    for i, dist in enumerate(distances_second_batch[0][:-1]):
        # assert sorting by score
        assert(distances_second_batch[0][i] < distances_second_batch[0][i+1])
        # assert that every distance in the second batch is higher than any distance of the first batch
        assert(len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) == 0)

    # reset
    batch_iterator.reset()

    # Run again in batches until depleted
    batch_size = 1500
    returned_results_num = 0
    iterations = 0
    start = time.time()
    while batch_iterator.has_next():
        iterations += 1
        labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
        returned_results_num += len(labels[0])

    test_logger.info(f'Total search time for running batches of size {batch_size} for index with {num_elements} of dim={dim}: {time.time() - start}')
    assert (returned_results_num == num_elements)
    assert (iterations == np.ceil(num_elements/batch_size))


def test_range_query(test_logger):
    dim = 128
    num_elements = 1000000

    bfparams = BFParams()
    bfparams.blockSize = num_elements
    bfparams.dim = dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_L2
    bfindex = BFIndex(bfparams)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []
    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))

    radius = 13
    start = time.time()
    bf_labels, bf_distances = bfindex.range_query(query_data, radius=radius)
    end = time.time()
    res_num = len(bf_labels[0])
    test_logger.info(f'lookup time for {num_elements} vectors with dim={dim} took {end - start} seconds, got {res_num} results')

    # Verify that we got exactly all vectors within the range
    dists = sorted([(spatial.distance.euclidean(query_data.flat, vec), key) for key, vec in vectors])
    keys = [key for _, key in dists[:res_num]]
    assert np.array_equal(np.array(bf_labels[0]), np.array(keys))
    # The distance return by the library is L2^2
    assert_allclose(max(bf_distances[0]), dists[res_num-1][0]**2, rtol=1e-05)
    assert max(bf_distances[0]) <= radius
    # Verify that the next closest vector that hasn't returned is not within the range
    assert dists[res_num][0]**2 > radius

    # Expect zero results for radius==0
    bf_labels, bf_distances = bfindex.range_query(query_data, radius=0)
    assert len(bf_labels[0]) == 0


def test_bf_multivalue(test_logger):
    dim = 128
    num_labels = 50000
    num_per_label = 20
    k=10

    num_elements = num_labels * num_per_label

    bfparams = BFParams()
    bfparams.blockSize = num_elements
    bfparams.dim = dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_Cosine
    bfparams.multi = True
    bfindex = BFIndex(bfparams)

    np.random.seed(47)
    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i % num_labels)
        vectors.append((i % num_labels, vector))

    query_data = np.float32(np.random.random((1, dim)))

    dists = {}
    for key, vec in vectors:
        # Setting or updating the score for each label. If it's the first time we calculate a score for a label,
        # dists.get(key, 3) will return 3, which is more than a Cosine score can be,
        # so we will choose the actual score the first time.
        dists[key] = min(spatial.distance.cosine(query_data.flat, vec), dists.get(key, 3)) # cosine distance is always <= 2

    dists = list(dists.items())
    dists = sorted(dists, key=lambda pair: pair[1])[:k]
    keys = [key for key, _ in dists[:k]]
    dists = [dist for _, dist in dists[:k]]
    start = time.time()
    bf_labels, bf_distances = bfindex.knn_query(query_data, k=10)
    end = time.time()
    test_logger.info(f'lookup time for {num_elements} vectors ({num_labels} labels and {num_per_label} vectors per label) with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)

def test_multi_range_query(test_logger):
    dim = 128
    num_labels = 20000
    per_label = 5

    num_elements = num_labels * per_label

    bfparams = BFParams()
    bfparams.dim = dim
    bfparams.metric = VecSimMetric_L2
    bfparams.multi = True
    bfparams.type = VecSimType_FLOAT32

    bfindex = BFIndex(bfparams)

    np.random.seed(47)
    data = np.float32(np.random.random((num_labels, per_label, dim)))
    vectors = []
    for label, vecs in enumerate(data):
        for vector in vecs:
            bfindex.add_vector(vector, label)
            vectors.append((label, vector))

    query_data = np.float32(np.random.random((1, dim)))

    radius = 13.0
    # calculate distances of the labels in the index
    dists = {}
    for key, vec in vectors:
        dists[key] = min(spatial.distance.sqeuclidean(query_data.flat, vec), dists.get(key, np.inf))

    dists = list(dists.items())
    dists = sorted(dists, key=lambda pair: pair[1])
    keys = [key for key, dist in dists if dist <= radius]

    start = time.time()
    bf_labels, bf_distances = bfindex.range_query(query_data, radius=radius)
    end = time.time()
    res_num = len(bf_labels[0])

    test_logger.info(f'lookup time for ({num_labels} X {per_label}) vectors with dim={dim} took {end - start} seconds')

    # Recall should be 100%.
    assert res_num == len(keys)

    # Compare the number of vectors that are actually within the range to the returned results.
    assert np.all(np.isin(bf_labels, np.array(keys)))

    # Asserts that all the results are unique
    assert len(bf_labels[0]) == len(np.unique(bf_labels[0]))

    assert max(bf_distances[0]) <= radius

    # Expect zero results for radius==0
    bf_labels, bf_distances = bfindex.range_query(query_data, radius=0)
    assert len(bf_labels[0]) == 0

class TestBfloat16():

    num_labels=10_000
    num_per_label=1
    dim = 128
    data = Data(VecSimType_BFLOAT16, VecSimMetric_L2, spatial.distance.sqeuclidean, vec_to_bfloat16, dim, num_labels, num_per_label)

    # Not testing bfloat16 cosine as type conversion biases mess up the results
    def test_bf_bfloat16_L2(self, test_logger):
        k = 10

        keys, dists = self.data.measure_dists(k)
        bf_labels, bf_distances = self.data.index.knn_query(self.data.query, k=k)
        assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
        assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)
        test_logger.info(f"sanity test for {self.data.metric} and {self.data.type} pass")

    def test_bf_bfloat16_batch_iterator(self, test_logger):
        bfindex = self.data.index
        num_elements = self.num_labels

        batch_iterator = bfindex.create_batch_iterator(self.data.query)
        labels_first_batch, distances_first_batch = batch_iterator.get_next_results(10, BY_ID)
        for i, _ in enumerate(labels_first_batch[0][:-1]):
            # assert sorting by id
            assert(labels_first_batch[0][i] < labels_first_batch[0][i+1])

        _, distances_second_batch = batch_iterator.get_next_results(10, BY_SCORE)
        for i, dist in enumerate(distances_second_batch[0][:-1]):
            # assert sorting by score
            assert(distances_second_batch[0][i] < distances_second_batch[0][i+1])
            # assert that every distance in the second batch is higher than any distance of the first batch
            assert(len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) == 0)

        # reset
        batch_iterator.reset()

        # Run again in batches until depleted
        batch_size = 1500
        returned_results_num = 0
        iterations = 0
        start = time.time()
        while batch_iterator.has_next():
            iterations += 1
            labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
            returned_results_num += len(labels[0])

        test_logger.info(f'Total search time for running batches of size {batch_size} for index with {num_elements} of dim={self.dim}: {time.time() - start}')
        assert (returned_results_num == num_elements)
        assert (iterations == np.ceil(num_elements/batch_size))

    def test_bf_bfloat16_range_query(self, test_logger):
        bfindex = self.data.index
        query_data = self.data.query

        radius = 14
        start = time.time()
        bf_labels, bf_distances = bfindex.range_query(query_data, radius=radius)
        end = time.time()
        res_num = len(bf_labels[0])
        test_logger.info(f'lookup time for {self.num_labels} vectors with dim={self.dim} took {end - start} seconds, got {res_num} results')

        # Verify that we got exactly all vectors within the range
        results, keys = get_ground_truth_results(spatial.distance.sqeuclidean, query_data.flat, self.data.vectors, res_num)

        assert_allclose(max(bf_distances[0]), results[res_num-1]["dist"], rtol=1e-05)
        assert np.array_equal(np.array(bf_labels[0]), np.array(keys))
        assert max(bf_distances[0]) <= radius
        # Verify that the next closest vector that hasn't returned is not within the range
        assert results[res_num]["dist"] > radius

        # Expect zero results for radius==0
        bf_labels, bf_distances = bfindex.range_query(query_data, radius=0)
        assert len(bf_labels[0]) == 0

def test_bf_bfloat16_multivalue(test_logger):
    num_labels=5_000
    num_per_label=20
    num_elements = num_labels * num_per_label

    dim = 128

    data = Data(VecSimType_BFLOAT16, VecSimMetric_L2, spatial.distance.sqeuclidean, vec_to_bfloat16, dim, num_labels, num_per_label)

    k=10

    query_data = data.query
    dists = {}
    for key, vec in data.vectors:
        # Setting or updating the score for each label.
        # If it's the first time we calculate a score for a label dists.get(key, dist)
        # will return dist so we will choose the actual score the first time.
        dist = spatial.distance.sqeuclidean(query_data.flat, vec)
        dists[key] = min(dist, dists.get(key, dist))

    dists = list(dists.items())
    dists = sorted(dists, key=lambda pair: pair[1])[:k]
    keys = [key for key, _ in dists[:k]]
    dists = [dist for _, dist in dists[:k]]

    start = time.time()
    bf_labels, bf_distances = data.index.knn_query(query_data, k=10)
    end = time.time()

    test_logger.info(f'lookup time for {num_elements} vectors ({num_labels} labels and {num_per_label} vectors per label) with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)

class TestFloat16():

    num_labels=10_000
    num_per_label=1
    dim = 128
    data = Data(VecSimType_FLOAT16, VecSimMetric_L2, spatial.distance.sqeuclidean, vec_to_float16, dim, num_labels, num_per_label)

    # Not testing bfloat16 cosine as type conversion biases mess up the results
    def test_bf_float16_L2(self, test_logger):
        k = 10

        keys, dists = self.data.measure_dists(k)
        bf_labels, bf_distances = self.data.index.knn_query(self.data.query, k=k)
        assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
        assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)
        test_logger.info(f"sanity test for {self.data.metric} and {self.data.type} pass")

    def test_bf_float16_batch_iterator(self, test_logger):
        bfindex = self.data.index
        num_elements = self.num_labels

        batch_iterator = bfindex.create_batch_iterator(self.data.query)
        labels_first_batch, distances_first_batch = batch_iterator.get_next_results(10, BY_ID)
        for i, _ in enumerate(labels_first_batch[0][:-1]):
            # assert sorting by id
            assert(labels_first_batch[0][i] < labels_first_batch[0][i+1])

        _, distances_second_batch = batch_iterator.get_next_results(10, BY_SCORE)
        for i, dist in enumerate(distances_second_batch[0][:-1]):
            # assert sorting by score
            assert(distances_second_batch[0][i] < distances_second_batch[0][i+1])
            # assert that every distance in the second batch is higher than any distance of the first batch
            assert(len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) == 0)

        # reset
        batch_iterator.reset()

        # Run again in batches until depleted
        batch_size = 1500
        returned_results_num = 0
        iterations = 0
        start = time.time()
        while batch_iterator.has_next():
            iterations += 1
            labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
            returned_results_num += len(labels[0])

        test_logger.info(f'Total search time for running batches of size {batch_size} for index with {num_elements} of dim={self.dim}: {time.time() - start}')
        assert (returned_results_num == num_elements)
        assert (iterations == np.ceil(num_elements/batch_size))

    def test_bf_float16_range_query(self, test_logger):
        bfindex = self.data.index
        query_data = self.data.query

        radius = 14
        start = time.time()
        bf_labels, bf_distances = bfindex.range_query(query_data, radius=radius)
        end = time.time()
        res_num = len(bf_labels[0])
        test_logger.info(f'lookup time for {self.num_labels} vectors with dim={self.dim} took {end - start} seconds, got {res_num} results')

        # Verify that we got exactly all vectors within the range
        results, keys = get_ground_truth_results(spatial.distance.sqeuclidean, query_data.flat, self.data.vectors, res_num)

        assert_allclose(max(bf_distances[0]), results[res_num-1]["dist"], rtol=1e-05)
        assert np.array_equal(np.array(bf_labels[0]), np.array(keys))
        assert max(bf_distances[0]) <= radius
        # Verify that the next closest vector that hasn't returned is not within the range
        assert results[res_num]["dist"] > radius

        # Expect zero results for radius==0
        bf_labels, bf_distances = bfindex.range_query(query_data, radius=0)
        assert len(bf_labels[0]) == 0

def test_bf_float16_multivalue(test_logger):
    num_labels=5_000
    num_per_label=20
    num_elements = num_labels * num_per_label

    dim = 128

    data = Data(VecSimType_FLOAT16, VecSimMetric_L2, spatial.distance.sqeuclidean, vec_to_float16, dim, num_labels, num_per_label)

    k=10

    query_data = data.query
    dists = {}
    for key, vec in data.vectors:
        # Setting or updating the score for each label.
        # If it's the first time we calculate a score for a label dists.get(key, dist)
        # will return dist so we will choose the actual score the first time.
        dist = spatial.distance.sqeuclidean(query_data.flat, vec)
        dists[key] = min(dist, dists.get(key, dist))

    dists = list(dists.items())
    dists = sorted(dists, key=lambda pair: pair[1])[:k]
    keys = [key for key, _ in dists[:k]]
    dists = [dist for _, dist in dists[:k]]

    start = time.time()
    bf_labels, bf_distances = data.index.knn_query(query_data, k=10)
    end = time.time()

    test_logger.info(f'lookup time for {num_elements} vectors ({num_labels} labels and {num_per_label} vectors per label) with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)

'''
A Class to run common tests for BF index

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
    dim = 128
    num_elements = 10_000
    num_queries = 1

    data_type = None

    rng = np.random.default_rng(seed=42)
    vectors_data = None
    query_data = None

    # single FLAT index with L2 metric
    cache_flat_index_L2_single = None
    cached_label_to_vec_list = None

    @classmethod
    def create_index(cls, metric = VecSimMetric_L2, is_multi=False):
        assert cls.data_type is not None
        return create_flat_index(cls.dim, metric, cls.data_type, is_multi=is_multi)

    @classmethod
    def create_add_vectors(cls, index):
        assert cls.vectors_data is not None
        return create_add_vectors(index, cls.vectors_data)

    @classmethod
    def get_cached_single_L2_index(cls):
        if cls.cache_flat_index_L2_single is None:
            cls.cache_flat_index_L2_single = cls.create_index()
            cls.cached_label_to_vec_list = cls.create_add_vectors(cls.cache_flat_index_L2_single)
        return cls.cache_flat_index_L2_single, cls.cached_label_to_vec_list

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
    def knn(cls, index, label_vec_list, dist_func, test_logger=logging.getLogger(__name__)):
        k = 10

        results, keys = get_ground_truth_results(dist_func, cls.query_data[0], label_vec_list, k)
        dists = [res["dist"] for res in results]
        bf_labels, bf_distances = index.knn_query(cls.query_data, k=k)
        assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
        assert_allclose(bf_distances, [dists[:k]],  rtol=1e-5, atol=0)
        test_logger.info(f"sanity test for L2 and {cls.data_type} pass")

    def test_L2(self, test_logger):
        index, label_to_vec_list = self.get_cached_single_L2_index()
        self.knn(index, label_to_vec_list, spatial.distance.sqeuclidean, test_logger)

    def test_batch_iterator(self, test_logger):
        index, _ = self.get_cached_single_L2_index()
        # num_elements = self.num_labels
        batch_size = 10


        batch_iterator = index.create_batch_iterator(self.query_data)
        labels_first_batch, distances_first_batch = batch_iterator.get_next_results(batch_size, BY_ID)
        for i, _ in enumerate(labels_first_batch[0][:-1]):
            # assert sorting by id
            assert(labels_first_batch[0][i] < labels_first_batch[0][i+1])

        _, distances_second_batch = batch_iterator.get_next_results(batch_size, BY_SCORE)
        for i, dist in enumerate(distances_second_batch[0][:-1]):
            # assert sorting by score
            assert(distances_second_batch[0][i] < distances_second_batch[0][i+1])
            # assert that every distance in the second batch is higher than any distance of the first batch
            assert(len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) == 0)

        # reset
        batch_iterator.reset()

        # Run again in batches until depleted
        batch_size = 1500
        returned_results_num = 0
        iterations = 0
        start = time.time()
        while batch_iterator.has_next():
            iterations += 1
            labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
            returned_results_num += len(labels[0])

        test_logger.info(f'Total search time for running batches of size {batch_size} for index with {self.num_elements} of dim={self.dim}: {time.time() - start}')
        assert (returned_results_num == self.num_elements)
        assert (iterations == np.ceil(self.num_elements/batch_size))

    ##### Should be explicitly called #####
    def range_query(self, dist_func, test_logger=logging.getLogger(__name__)):
        bfindex = self.create_index(VecSimMetric_Cosine)
        label_to_vec_list = self.create_add_vectors(bfindex)
        radius = bfindex.knn_query(self.query_data[0], k=100)[1][0][-1] # get the distance of the 100th closest vector as the radius

        start = time.time()
        bf_labels, bf_distances = bfindex.range_query(self.query_data[0], radius=radius)
        end = time.time()
        res_num = len(bf_labels[0])
        test_logger.info(f'lookup time for {self.num_elements} vectors with dim={self.dim} took {end - start} seconds, got {res_num} results')


        # Verify that we got exactly all vectors within the range
        results, keys = get_ground_truth_results(dist_func, self.query_data[0], label_to_vec_list, res_num)

        assert_allclose(bf_distances[0], [results[i]["dist"] for i in range(res_num)], rtol=1e-05)
        assert self.compute_correct(bf_labels[0], bf_distances[0], keys, results) == res_num
        assert max(bf_distances[0]) <= radius
        # Verify that the next closest vector that hasn't returned is not within the range
        assert results[res_num]["dist"] > radius

        # Expect zero results for radius==0
        bf_labels, bf_distances = bfindex.range_query(self.query_data[0], radius=0)
        assert len(bf_labels[0]) == 0

    def multi_value(self, create_data_func, num_per_label = 5, test_logger=logging.getLogger(__name__)):
        # num_labels=5_000
        # num_per_label=20
        # num_elements = num_labels * num_per_label
        num_labels = self.num_elements // num_per_label
        k = 10

        data = create_data_func((num_labels, num_per_label, self.dim), self.rng)

        index = self.create_index(is_multi=True)

        vectors = []
        for i, cur_vectors in enumerate(data):
            for vector in cur_vectors:
                index.add_vector(vector, i)
                vectors.append((i, vector))

        dists = {}
        for key, vec in vectors:
            # Setting or updating the score for each label.
            # If it's the first time we calculate a score for a label dists.get(key, dist)
            # will return dist so we will choose the actual score the first time.
            dist = spatial.distance.sqeuclidean(self.query_data[0], vec)
            dists[key] = min(dist, dists.get(key, dist))

        dists = list(dists.items())
        dists = sorted(dists, key=lambda pair: pair[1])[:k]
        keys = [key for key, _ in dists[:k]]
        dists = [dist for _, dist in dists[:k]]

        start = time.time()
        bf_labels, bf_distances = index.knn_query(self.query_data[0], k=k)
        end = time.time()

        test_logger.info(f'lookup time for {self.num_elements} vectors ({num_labels} labels and {num_per_label} vectors per label) with dim={self.dim} took {end - start} seconds')

        assert_allclose(bf_labels, [keys],  rtol=1e-5)
        assert_allclose(bf_distances, [dists],  rtol=1e-5)

class TestINT8(GeneralTest):

    data_type = VecSimType_INT8

    #### Create vectors
    vectors_data = create_int8_vectors((GeneralTest.num_elements, GeneralTest.dim), GeneralTest.rng)

    #### Create queries
    query_data = create_int8_vectors((GeneralTest.num_queries, GeneralTest.dim), GeneralTest.rng)

    def test_Cosine(self):

        index = self.create_index(VecSimMetric_Cosine)
        label_to_vec_list = self.create_add_vectors(index)

        self.knn(index, label_to_vec_list, fp32_expand_and_calc_cosine_dist)

    def test_range_query(self):
        self.range_query(fp32_expand_and_calc_cosine_dist)

    def test_multi_value(self):
        self.multi_value(create_int8_vectors)

class TestUINT8(GeneralTest):

    data_type = VecSimType_UINT8

    #### Create vectors
    vectors_data = create_uint8_vectors((GeneralTest.num_elements, GeneralTest.dim), GeneralTest.rng)

    #### Create queries
    query_data = create_uint8_vectors((GeneralTest.num_queries, GeneralTest.dim), GeneralTest.rng)

    def test_Cosine(self):

        index = self.create_index(VecSimMetric_Cosine)
        label_to_vec_list = self.create_add_vectors(index)

        self.knn(index, label_to_vec_list, fp32_expand_and_calc_cosine_dist)

    def test_range_query(self):
        self.range_query(fp32_expand_and_calc_cosine_dist)

    def test_multi_value(self):
        self.multi_value(create_uint8_vectors)
