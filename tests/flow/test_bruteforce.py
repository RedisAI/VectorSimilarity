from common import *


def test_bf_cosine():
    dim = 128
    num_elements = 1000000
    k=10

    bfparams = BFParams()
    bfparams.initialCapacity = num_elements
    bfparams.blockSize = num_elements
    bfindex = BFIndex(bfparams, VecSimType_FLOAT32, dim, VecSimMetric_Cosine)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))

    dists = [(spatial.distance.cosine(query_data, vec), key) for key, vec in vectors]
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
    bfindex = BFIndex(bfparams, VecSimType_FLOAT32, dim, VecSimMetric_L2)

    data = np.float32(np.random.random((num_elements, dim)))
    vectors = []

    for i, vector in enumerate(data):
        bfindex.add_vector(vector, i)
        vectors.append((i, vector))

    query_data = np.float32(np.random.random((1, dim)))

    dists = [(spatial.distance.euclidean(query_data, vec), key) for key, vec in vectors]
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
    bf_params = BFParams()
    bf_params.initialCapacity = num_elements
    bf_params.blockSize = num_elements
    bf_index = BFIndex(bf_params, VecSimType_FLOAT32, dim, VecSimMetric_L2)

    # Add 1M random vectors to the index
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
