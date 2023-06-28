# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

from common import *

def test_sanity_bf():
    class TestData:
        def __init__(self, data_type, metric, dist_func, np_fuc):
            dim = 16
            num_elements = 10
            bfparams = BFParams()

            bfparams.initialCapacity = num_elements
            bfparams.blockSize = num_elements
            bfparams.dim = dim
            bfparams.type = data_type
            bfparams.metric = metric

            self.index = BFIndex(bfparams)

            self.metric = metric
            self.type = data_type
            self.dist_func = dist_func

            np.random.seed(47)
            self.data = np_fuc(np.random.random((num_elements, dim)))
            self.query = np_fuc(np.random.random((1, dim)))
            self.vectors = []
            for i, vector in enumerate(self.data):
                self.vectors.append((i, vector))
                self.index.add_vector(vector, i)

        def measure_dists(self, k):
            dists = [(self.dist_func(self.query.flat, vec), key) for key, vec in self.vectors]
            dists = sorted(dists)[:k]
            keys = [key for _, key in dists]
            dists = [dist for dist, _ in dists]
            return (keys, dists)       
    
    test_datas = []

    dist_funcs = [(VecSimMetric_Cosine, spatial.distance.cosine), (VecSimMetric_L2, spatial.distance.sqeuclidean)]
    types = [(VecSimType_FLOAT32, np.float32), (VecSimType_FLOAT64, np.float64)]
    for type_name, np_type in types:
        for dist_name, dist_func in dist_funcs:
            test_datas.append(TestData(type_name, dist_name, dist_func, np_type))

    k = 10
    for test_data in test_datas:

        keys, dists = test_data.measure_dists(k)
        bf_labels, bf_distances = test_data.index.knn_query(test_data.query, k=k)
        assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
        assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)
        print(f"\nsanity test for {test_data.metric} and {test_data.type} pass")

def test_bf_cosine():
    dim = 128
    num_elements = 1000000
    k=10

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
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
    print(f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)


def test_bf_l2():
    dim = 128
    num_elements = 1000000
    k=10

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
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
    print(f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)


def test_batch_iterator():
    dim = 128
    num_elements = 1000000

    # Create a brute force index for vectors of 128 floats. Use 'Cosine' as the distance metric
    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
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

    print(f'Total search time for running batches of size {batch_size} for index with {num_elements} of dim={dim}: {time.time() - start}')
    assert (returned_results_num == num_elements)
    assert (iterations == np.ceil(num_elements/batch_size))


def test_range_query():
    dim = 128
    num_elements = 1000000

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
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
    print(f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds, got {res_num} results')

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


def test_bf_multivalue():
    dim = 128
    num_labels = 50000
    num_per_label = 20
    k=10

    num_elements = num_labels * num_per_label

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
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
    print(f'\nlookup time for {num_elements} vectors ({num_labels} labels and {num_per_label} vectors per label) with dim={dim} took {end - start} seconds')

    assert_allclose(bf_labels, [keys],  rtol=1e-5, atol=0)
    assert_allclose(bf_distances, [dists],  rtol=1e-5, atol=0)

def test_multi_range_query():
    dim = 128
    num_labels = 20000
    per_label = 5

    num_elements = num_labels * per_label

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
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

    print(f'\nlookup time for ({num_labels} X {per_label}) vectors with dim={dim} took {end - start} seconds')

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
