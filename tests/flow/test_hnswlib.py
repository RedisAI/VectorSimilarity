import os
from common import *
import hnswlib
from redis import Redis


# # compare results with the original version of hnswlib - do not use elements deletion.
# def test_sanity_hnswlib_index_L2():
#     dim = 16
#     num_elements = 10000
#     space = 'l2'
#     M=16
#     efConstruction = 100

#     efRuntime = 10

#     params = VecSimParams()
#     hnswparams = HNSWParams()

#     params.algo = VecSimAlgo_HNSWLIB

#     hnswparams.dim = dim
#     hnswparams.metric = VecSimMetric_L2
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.M = M
#     hnswparams.efConstruction = efConstruction
#     hnswparams.initialCapacity = num_elements
#     hnswparams.efRuntime=efRuntime

#     params.hnswParams = hnswparams
#     index = VecSimIndex(params)

#     p = hnswlib.Index(space=space, dim=dim)
#     p.init_index(max_elements=num_elements, ef_construction=efConstruction, M=M)
#     p.set_ef(efRuntime)

#     data = np.float32(np.random.random((num_elements, dim)))
#     for i, vector in enumerate(data):
#         index.add_vector(vector, i)
#         p.add_items(vector, i)

#     query_data = np.float32(np.random.random((1, dim)))
#     hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
#     redis_labels, redis_distances = index.knn_query(query_data, 10)
#     assert_allclose(hnswlib_labels, redis_labels,  rtol=1e-5, atol=0)
#     assert_allclose(hnswlib_distances, redis_distances,  rtol=1e-5, atol=0)


# def test_sanity_hnswlib_index_cosine():
#     dim = 16
#     num_elements = 10000
#     space = 'cosine'
#     M=16
#     efConstruction = 100

#     efRuntime = 10

#     params = VecSimParams()
#     hnswparams = HNSWParams()

#     params.algo = VecSimAlgo_HNSWLIB

#     hnswparams.dim = dim
#     hnswparams.metric = VecSimMetric_Cosine
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.M = M
#     hnswparams.efConstruction = efConstruction
#     hnswparams.initialCapacity = num_elements
#     hnswparams.efRuntime=efRuntime

#     params.hnswParams = hnswparams
#     index = VecSimIndex(params)

#     p = hnswlib.Index(space=space, dim=dim)
#     p.init_index(max_elements=num_elements, ef_construction=efConstruction, M=M)
#     p.set_ef(efRuntime)

#     data = np.float32(np.random.random((num_elements, dim)))
#     for i, vector in enumerate(data):
#         index.add_vector(vector, i)
#         p.add_items(vector, i)

#     query_data = np.float32(np.random.random((1, dim)))
#     hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
#     redis_labels, redis_distances = index.knn_query(query_data, 10)
#     assert_allclose(hnswlib_labels, redis_labels,  rtol=1e-5, atol=0)
#     assert_allclose(hnswlib_distances, redis_distances,  rtol=1e-5, atol=0)


# # Validate correctness of delete implementation comparing the brute force search. We test the search recall which is not
# # deterministic, but should be above a certain threshold. Note that recall is highly impacted by changing
# # index parameters.
# def test_recall_for_hnswlib_index_with_deletion():
#     dim = 16
#     num_elements = 10000
#     M = 16
#     efConstruction = 100

#     num_queries = 10
#     k=10
#     efRuntime = 0

#     hnswparams = HNSWParams()
#     hnswparams.M = M
#     hnswparams.efConstruction = efConstruction
#     hnswparams.initialCapacity = num_elements
#     hnswparams.efRuntime = efRuntime
#     hnswparams.dim = dim
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.metric = VecSimMetric_L2

#     hnsw_index = HNSWIndex(hnswparams)

#     data = np.float32(np.random.random((num_elements, dim)))
#     vectors = []
#     for i, vector in enumerate(data):
#         hnsw_index.add_vector(vector, i)
#         vectors.append((i, vector))

#     # delete half of the data
#     for i in range(0, len(data), 2):
#         hnsw_index.delete_vector(i)
#     vectors = [vectors[i] for i in range(1, len(data), 2)]

#     # We validate that we can increase ef with this designated API (if this won't work, recall should be very low)
#     hnsw_index.set_ef(50)
#     query_data = np.float32(np.random.random((num_queries, dim)))
#     correct = 0
#     for target_vector in query_data:
#         hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

#         # sort distances of every vector from the target vector and get actual k nearest vectors
#         dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
#         dists = sorted(dists)
#         keys = [key for _, key in dists[:k]]

#         for label in hnswlib_labels[0]:
#             for correct_label in keys:
#                 if label == correct_label:
#                     correct+=1
#                     break

#     # Measure recall
#     recall = float(correct)/(k*num_queries)
#     print("\nrecall is: \n", recall)
#     assert(recall > 0.9)


# def test_batch_iterator():
#     dim = 100
#     num_elements = 100000
#     M = 26
#     efConstruction = 180
#     efRuntime = 180
#     num_queries = 10

#     hnswparams = HNSWParams()
#     hnswparams.M = M
#     hnswparams.efConstruction = efConstruction
#     hnswparams.initialCapacity = num_elements
#     hnswparams.efRuntime = efRuntime
#     hnswparams.dim = dim
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.metric = VecSimMetric_L2

#     hnsw_index = HNSWIndex(hnswparams)

#     # Add 100k random vectors to the index
#     rng = np.random.default_rng(seed=47)
#     data = np.float32(rng.random((num_elements, dim)))
#     vectors = []
#     for i, vector in enumerate(data):
#         hnsw_index.add_vector(vector, i)
#         vectors.append((i, vector))

#     # Create a random query vector and create a batch iterator
#     query_data = np.float32(rng.random((1, dim)))
#     batch_iterator = hnsw_index.create_batch_iterator(query_data)
#     labels_first_batch, distances_first_batch = batch_iterator.get_next_results(10, BY_ID)
#     for i, _ in enumerate(labels_first_batch[0][:-1]):
#         # Assert sorting by id
#         assert(labels_first_batch[0][i] < labels_first_batch[0][i+1])

#     labels_second_batch, distances_second_batch = batch_iterator.get_next_results(10, BY_SCORE)
#     should_have_return_in_first_batch = []
#     for i, dist in enumerate(distances_second_batch[0][:-1]):
#         # Assert sorting by score
#         assert(distances_second_batch[0][i] < distances_second_batch[0][i+1])
#         # Assert that every distance in the second batch is higher than any distance of the first batch
#         if len(distances_first_batch[0][np.where(distances_first_batch[0] > dist)]) != 0:
#             should_have_return_in_first_batch.append(dist)
#     assert(len(should_have_return_in_first_batch) <= 2)

#     # Verify that runtime args are sent properly to the batch iterator.
#     query_params = VecSimQueryParams()
#     query_params.hnswRuntimeParams.efRuntime = 5
#     batch_iterator_new = hnsw_index.create_batch_iterator(query_data, query_params)
#     labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
#     # Verify that accuracy is worse with the new lower ef_runtime.
#     assert (sum(labels_first_batch[0]) < sum(labels_first_batch_new[0]))

#     query_params.hnswRuntimeParams.efRuntime = efRuntime  # Restore previous ef_runtime.
#     batch_iterator_new = hnsw_index.create_batch_iterator(query_data, query_params)
#     labels_first_batch_new, distances_first_batch_new = batch_iterator_new.get_next_results(10, BY_ID)
#     # Verify that results are now the same.
#     assert_allclose(labels_first_batch_new[0], labels_first_batch[0])

#     # Reset
#     batch_iterator.reset()

#     # Run in batches of 100 until we reach 1000 results and measure recall
#     batch_size = 100
#     total_res = 1000
#     total_recall = 0
#     query_data = np.float32(rng.random((num_queries, dim)))
#     for target_vector in query_data:
#         correct = 0
#         batch_iterator = hnsw_index.create_batch_iterator(target_vector)
#         iterations = 0
#         # Sort distances of every vector from the target vector and get the actual order
#         dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
#         dists = sorted(dists)
#         accumulated_labels = []
#         while batch_iterator.has_next():
#             iterations += 1
#             labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
#             accumulated_labels.extend(labels[0])
#             returned_results_num = len(accumulated_labels)
#             if returned_results_num == total_res:
#                 keys = [key for _, key in dists[:returned_results_num]]
#                 correct += len(set(accumulated_labels).intersection(set(keys)))
#                 break
#         assert iterations == np.ceil(total_res/batch_size)
#         recall = float(correct) / total_res
#         assert recall >= 0.89
#         total_recall += recall
#     print(f'\nAvg recall for {total_res} results in index of size {num_elements} with dim={dim} is: ', total_recall/num_queries)

#     # Run again a single query in batches until it is depleted.
#     batch_iterator = hnsw_index.create_batch_iterator(query_data[0])
#     iterations = 0
#     accumulated_labels = set()

#     while batch_iterator.has_next():
#         iterations += 1
#         labels, distances = batch_iterator.get_next_results(batch_size, BY_SCORE)
#         # Verify that we got new scores in each iteration.
#         assert len(accumulated_labels.intersection(set(labels[0]))) == 0
#         accumulated_labels = accumulated_labels.union(set(labels[0]))
#     assert len(accumulated_labels) >= 0.95*num_elements
#     print("Overall results returned:", len(accumulated_labels), "in", iterations, "iterations")


# def test_serialization():
#     dim = 16
#     num_elements = 10000
#     M = 16
#     efConstruction = 100

#     num_queries = 10
#     k = 10
#     efRuntime = 50

#     hnswparams = HNSWParams()
#     hnswparams.M = M
#     hnswparams.efConstruction = efConstruction
#     hnswparams.initialCapacity = num_elements
#     hnswparams.efRuntime = efRuntime
#     hnswparams.dim = dim
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.metric = VecSimMetric_L2

#     hnsw_index = HNSWIndex(hnswparams)

#     data = np.float32(np.random.random((num_elements, dim)))
#     vectors = []
#     for i, vector in enumerate(data):
#         hnsw_index.add_vector(vector, i)
#         vectors.append((i, vector))

#     query_data = np.float32(np.random.random((num_queries, dim)))
#     correct = 0
#     correct_labels = []  # cache these
#     for target_vector in query_data:
#         hnswlib_labels, hnswlib_distances = hnsw_index.knn_query(target_vector, 10)

#         # sort distances of every vector from the target vector and get actual k nearest vectors
#         dists = [(spatial.distance.euclidean(target_vector, vec), key) for key, vec in vectors]
#         dists = sorted(dists)
#         keys = [key for _, key in dists[:k]]
#         correct_labels.append(keys)

#         for label in hnswlib_labels[0]:
#             for correct_label in keys:
#                 if label == correct_label:
#                     correct += 1
#                     break
#     # Measure recall
#     recall = float(correct)/(k*num_queries)
#     print("\nrecall is: \n", recall)

#     # Persist, delete and restore index.
#     file_name = os.getcwd()+"/dump"
#     hnsw_index.save_index(file_name)

#     new_hnsw_index = HNSWIndex(hnswparams)
#     assert new_hnsw_index.index_size() == 0
#     new_hnsw_index.load_index(file_name)
#     os.remove(file_name)
#     assert new_hnsw_index.index_size() == num_elements

#     # Check recall
#     correct_after = 0
#     for i, target_vector in enumerate(query_data):
#         hnswlib_labels, hnswlib_distances = new_hnsw_index.knn_query(target_vector, 10)
#         correct_labels_cur = correct_labels[i]
#         for label in hnswlib_labels[0]:
#             for correct_label in correct_labels_cur:
#                 if label == correct_label:
#                     correct_after += 1
#                     break

#     # Compare recall after reloading the index
#     recall_after = float(correct_after)/(k*num_queries)
#     print("\nrecall after is: \n", recall_after)
#     assert recall == recall_after


# def test_range_query():
#     dim = 100
#     num_elements = 100000
#     epsilon = 0.01

#     params = VecSimParams()
#     hnswparams = HNSWParams()

#     params.algo = VecSimAlgo_HNSWLIB

#     hnswparams.dim = dim
#     hnswparams.metric = VecSimMetric_L2
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.M = 32
#     hnswparams.efConstruction = 200
#     hnswparams.initialCapacity = num_elements
#     hnswparams.epsilon = epsilon

#     params.hnswParams = hnswparams
#     index = VecSimIndex(params)

#     np.random.seed(47)
#     data = np.float32(np.random.random((num_elements, dim)))
#     vectors = []
#     for i, vector in enumerate(data):
#         index.add_vector(vector, i)
#         vectors.append((i, vector))

#     query_data = np.float32(np.random.random((1, dim)))

#     radius = 13.0
#     recalls = {}

#     for epsilon_rt in [0.001, 0.01, 0.1]:
#         query_params = VecSimQueryParams()
#         query_params.hnswRuntimeParams.epsilon = epsilon_rt
#         start = time.time()
#         hnsw_labels, hnsw_distances = index.range_query(query_data, radius=radius, query_param=query_params)
#         end = time.time()
#         res_num = len(hnsw_labels[0])

#         dists = sorted([(key, spatial.distance.sqeuclidean(query_data, vec)) for key, vec in vectors])
#         actual_results = [(key, dist) for key, dist in dists if dist <= radius]

#         print(f'\nlookup time for {num_elements} vectors with dim={dim} took {end - start} seconds with epsilon={epsilon_rt},'
#               f' got {res_num} results, which are {res_num/len(actual_results)} of the entire results in the range.')

#         # Compare the number of vectors that are actually within the range to the returned results.
#         assert np.all(np.isin(hnsw_labels, np.array([label for label, _ in actual_results])))

#         assert max(hnsw_distances[0]) <= radius
#         recalls[epsilon_rt] = res_num/len(actual_results)

#     # Expect higher recalls for higher epsilon values.
#     assert recalls[0.001] <= recalls[0.01] <= recalls[0.1]

#     # Expect zero results for radius==0
#     hnsw_labels, hnsw_distances = index.range_query(query_data, radius=0)
#     assert len(hnsw_labels[0]) == 0


def test_geo_range_query():
    dim = 2
    num_elements = 100000
    radius = 180000

    bfparams = BFParams()
    hnswparams = HNSWParams()

    hnswparams.dim = dim
    hnswparams.metric = VecSimMetric_WGS84
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.M = 8
    hnswparams.efConstruction = 100
    hnswparams.initialCapacity = num_elements

    bfparams.dim = dim
    bfparams.metric = VecSimMetric_WGS84
    bfparams.type = VecSimType_FLOAT32
    bfparams.initialCapacity = num_elements

    hnsw = HNSWIndex(hnswparams)
    bf = BFIndex(bfparams)
    conn = Redis()
    conn.flushall()
    conn.execute_command('FT.CREATE idx SCHEMA geo geo')

    np.random.seed(47)
    data = (360.0, 180.0)*np.random.random((num_elements, dim))-(180.0, 90.0)
    print('\ndata looks like:\n', data[:3])
    vectors = []
    for i, vector in enumerate(data):
        hnsw.add_vector(vector, i)
        bf.add_vector(vector, i)
        conn.hset(i, 'geo', f'{vector[0]},{vector[1]}')
        vectors.append((i, vector))

    query_data = np.float32([360.0*np.random.random()-180.0, 180.0*np.random.random()-90.0])
    print('\nquery is:\n', query_data)

    start = time.time()
    hnsw_labels, hnsw_distances = hnsw.range_query(query_data, radius)
    hnsw_time = (time.time() - start)
    start = time.time()
    bf_labels, bf_distances = bf.range_query(query_data, radius)
    bf_time = (time.time() - start)
    start = time.time()
    redis_res = conn.ft('idx').search(f'@geo:[{query_data[0]} {query_data[1]} {radius} m]')
    redis_time = (time.time() - start)

    redis_labels = [int(doc.id) for doc in redis_res.docs]

    print('\nFLAT results:', bf_labels[0], bf_distances[0], sep='\n')
    print('took:', bf_time*1000, 'milliseconds')
    print('\nHNSW results:', hnsw_labels[0], hnsw_distances[0], sep='\n')
    print('took:', hnsw_time*1000, 'milliseconds')
    print('\nFT.SEARCH results:', redis_labels, sep='\n')
    print('took:', redis_time*1000, 'milliseconds')

    # Measure recall
    recall = float(len([e for e in hnsw_labels[0] if e in bf_labels[0]]))/(len(bf_labels[0]))
    print("\nrecall is: \n", recall)
    assert(recall > 0.9)


# def test_geo_3d_range_query():
#     dim = 3
#     num_elements = 100000
#     k = 10

#     bfparams = BFParams()
#     hnswparams = HNSWParams()

#     hnswparams.dim = dim
#     hnswparams.metric = VecSimMetric_WGS84_3D
#     hnswparams.type = VecSimType_FLOAT32
#     hnswparams.M = 8
#     hnswparams.efConstruction = 100
#     hnswparams.initialCapacity = num_elements

#     bfparams.dim = dim
#     bfparams.metric = VecSimMetric_WGS84_3D
#     bfparams.type = VecSimType_FLOAT32
#     bfparams.initialCapacity = num_elements

#     hnsw = HNSWIndex(hnswparams)
#     bf = BFIndex(bfparams)

#     np.random.seed(47)
#     data = (360.0, 180.0, 1000.0)*np.random.random((num_elements, dim))-(180.0, 90.0, 0.0)
#     print('\ndata looks like:\n', data[0:3])
#     vectors = []
#     for i, vector in enumerate(data):
#         hnsw.add_vector(vector, i)
#         bf.add_vector(vector, i)
#         vectors.append((i, vector))

#     query_data = np.float32([360.0*np.random.random()-180.0, 180.0*np.random.random()-90.0, np.random.random()*1000.0])
#     print('\nquery is:\n', query_data)

#     start = time.time()
#     hnsw_labels, hnsw_distances = hnsw.knn_query(query_data, k)
#     hnsw_time = (time.time() - start)
#     start = time.time()
#     bf_labels, bf_distances = bf.knn_query(query_data, k)
#     bf_time = (time.time() - start)

#     print('\nFLAT results:', bf_labels[0], bf_distances[0], sep='\n')
#     print('took:', bf_time*1000, 'milliseconds')
#     print('\nHNSW results:', hnsw_labels[0], hnsw_distances[0], sep='\n')
#     print('took:', hnsw_time*1000, 'milliseconds')

#     # Measure recall
#     recall = float(len(set(hnsw_labels[0]).intersection(bf_labels[0])))/(k)
#     print("\nrecall is: \n", recall)
#     assert(recall > 0.9)
