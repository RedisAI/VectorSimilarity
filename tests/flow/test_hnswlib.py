from common import *
import hnswlib


# compare results with the original version of hnswlib - do not use elements deletion.
def test_sanity_hnswlib_index_L2():
    dim = 16
    num_elements = 10000
    space = 'l2'
    M=16
    efConstruction = 100

    efRuntime = 10

    params = VecSimParams()
    hnswparams = HNSWParams()

    params.algo = VecSimAlgo_HNSWLIB

    hnswparams.dim = dim
    hnswparams.metric = VecSimMetric_L2
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.M = M
    hnswparams.efConstruction = efConstruction
    hnswparams.initialCapacity = num_elements
    hnswparams.efRuntime=efRuntime

    params.hnswParams = hnswparams
    index = VecSimIndex(params)

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
    assert_allclose(hnswlib_labels, redis_labels,  rtol=1e-5, atol=0)
    assert_allclose(hnswlib_distances, redis_distances,  rtol=1e-5, atol=0)

def test_sanity_hnswlib_index_cosine():
    dim = 16
    num_elements = 10000
    space = 'cosine'
    M=16
    efConstruction = 100

    efRuntime = 10

    params = VecSimParams()
    hnswparams = HNSWParams()

    params.algo = VecSimAlgo_HNSWLIB

    hnswparams.dim = dim
    hnswparams.metric = VecSimMetric_Cosine
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.M = M
    hnswparams.efConstruction = efConstruction
    hnswparams.initialCapacity = num_elements
    hnswparams.efRuntime=efRuntime

    params.hnswParams = hnswparams
    index = VecSimIndex(params)

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
    assert_allclose(hnswlib_labels, redis_labels,  rtol=1e-5, atol=0)
    assert_allclose(hnswlib_distances, redis_distances,  rtol=1e-5, atol=0)



# Validate correctness of delete implementation comparing the brute force search. We test the search recall which is not
# deterministic, but should be above a certain threshold. Note that recall is highly impacted by changing
# index parameters.
def test_recall_for_hnswlib_index_with_deletion():
    dim = 16
    num_elements = 10000
    M = 16
    efConstruction = 100

    num_queries = 10
    k=10
    efRuntime = 0

    hnswparams = HNSWParams()
    hnswparams.M = M
    hnswparams.efConstruction = efConstruction
    hnswparams.initialCapacity = num_elements
    hnswparams.efRuntime = efRuntime
    hnswparams.dim = dim
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_L2

    hnsw_index = HNSWIndex(hnswparams)

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
                    correct+=1
                    break

    # Measure recall
    recall = float(correct)/(k*num_queries)
    print("\nrecall is: \n", recall)
    assert(recall > 0.9)


def test_batch_iterator():
    dim = 100
    num_elements = 100000
    M = 16
    efConstruction = 100
    efRuntime = 100

    num_queries = 10

    hnswparams = HNSWParams()
    hnswparams.M = M
    hnswparams.efConstruction = efConstruction
    hnswparams.initialCapacity = num_elements
    hnswparams.efRuntime = efRuntime
    hnswparams.dim = dim
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_L2

    hnsw_index = HNSWIndex(hnswparams)

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
    batch_size = 100
    total_res = 10000
    iterations = 0
    correct = 0
    query_data = np.float32(rng.random((num_queries, dim)))
    for target_vector in query_data:
        batch_iterator = hnsw_index.create_batch_iterator(target_vector)
        # sort distances of every vector from the target vector and get actual k nearest vectors
        dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
        dists = sorted(dists)
        accumulated_labels = []
        while batch_iterator.has_next():
            iterations += 1
            labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
            accumulated_labels.extend(labels[0])
            returned_results_num = len(accumulated_labels)
            if returned_results_num == total_res:
                print("measure recall")
                returned_results_num = len(accumulated_labels)
                keys = [key for _, key in dists[:returned_results_num]]
                correct += len(set(accumulated_labels).intersection(set(keys)))
                # for label in accumulated_labels:
                #     for correct_label in keys:
                #         if label == correct_label:
                #             correct += 1
                #             break
                # Measure iteration recall
                # recall = float(correct)/returned_results_num
                break
    recall = float(correct) / (total_res*num_queries)
    print(f'\nrecall is: ', recall)

    # print(f'Total search time for running batches of size {batch_size} for index with {num_elements} of dim={dim}: {time.time() - start}')
    # assert (returned_results_num == num_elements)
    # assert (iterations == np.ceil(num_elements/batch_size))
