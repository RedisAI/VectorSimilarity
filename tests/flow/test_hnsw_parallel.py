# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).
import concurrent
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait

from common import *

# Helper class for creating a "baseline" HNSW index that was built by inserting vectors one by one and a corresponding
# flat index for given params, to compare the parallel operation against it
class TestIndex:
    def __init__(self, dim_, num_elements_, metric_, data_type_, multi_=False, ef_runtime=200):

        self.dim = dim_
        self.num_elements = num_elements_
        self.metric = metric_
        self.data_type = data_type_
        self.hnsw_index = create_hnsw_index(dim_, num_elements_, metric_, data_type_,
                                            ef_runtime=ef_runtime, is_multi=multi_)
        self.multi = multi_

        bf_params = BFParams()

        bf_params.initialCapacity = num_elements_
        bf_params.blockSize = num_elements_
        bf_params.dim = dim_
        bf_params.type = data_type_
        bf_params.metric = metric_
        bf_params.multi = multi_

        self.bf_index = BFIndex(bf_params)

        np.random.seed(47)
        self.data = None
        self.total_res_bf = []  # Save the ground truth results
        self.sequential_insert_time = 0  # Total time took to insert vectors to the index one by one
        self.vectors_per_label = 1

    def insert_random_vectors(self):
        self.data = np.float32(np.random.random((self.num_elements, self.dim))) \
            if self.data_type == VecSimType_FLOAT32 else np.random.random((self.num_elements, self.dim))

        self.sequential_insert_time = 0
        for label, vector in enumerate(self.data):
            start = time.time()
            self.hnsw_index.add_vector(vector, label)
            self.sequential_insert_time += time.time() - start
            self.bf_index.add_vector(vector, label)

    def insert_random_vectors_multi(self, vec_per_label):
        self.vectors_per_label = vec_per_label
        self.data = np.float32(np.random.random((int(self.num_elements/per_label), per_label, self.dim))) \
            if self.data_type == VecSimType_FLOAT32\
            else np.random.random((int(self.num_elements/per_label), per_label, self.dim))

        self.sequential_insert_time = 0
        for label, vectors in enumerate(self.data):
            for vector in vectors:
                start = time.time()
                self.hnsw_index.add_vector(vector, label)
                self.sequential_insert_time += time.time() - start
                self.bf_index.add_vector(vector, label)

    def compute_ground_truth_knn(self, query_data, k):
        self.total_res_bf = []  # reset upon every call, so it will be aligned with the given query data
        for query in query_data:
            self.total_res_bf.append(self.bf_index.knn_query(query, k)[0][0])

    def compute_ground_truth_range(self, query_data, radius):
        self.total_res_bf = []  # reset upon every call, so it will be aligned with the given query data
        for query in query_data:
            self.total_res_bf.append(self.bf_index.range_query(query, radius)[0][0])


# Global test params
dim = 32
num_elements = 100000
metric = VecSimMetric_L2
data_type = VecSimType_FLOAT32
per_label = 5  # for multi value index

print("Creating test indexes...")
g_test_index = TestIndex(dim, num_elements, metric, data_type)
g_test_index.insert_random_vectors()

g_test_index_multi = TestIndex(dim, num_elements, metric, data_type, multi_=True)
g_test_index_multi.insert_random_vectors_multi(per_label)

def test_parallel_search():
    k = 10
    num_queries = 10000
    n_threads = min(os.cpu_count(), 8)
    expected_parallel_rate = 0.9  # we expect that at least 90% of the insert/search time will be executed in parallel

    # Sequential search as the baseline
    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index.compute_ground_truth_knn(query_data, k)
    total_search_time = 0
    total_correct = 0
    for i, query in enumerate(query_data):
        start = time.time()
        res_labels, _ = g_test_index.hnsw_index.knn_query(query, k)
        total_search_time += time.time() - start
        total_correct += len(set(res_labels[0]).intersection(set(g_test_index.total_res_bf[i])))

    print(f"Running sequential search, got {total_correct / (k * num_queries)} recall on {num_queries} queries,"
          f" and {num_queries/total_search_time} query per seconds")

    start = time.time()
    res_labels, _ = g_test_index.hnsw_index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time_parallel = time.time() - start

    total_correct_parallel = 0
    for i in range(num_queries):
        total_correct_parallel += len(set(g_test_index.total_res_bf[i]).intersection(set(res_labels[i])))

    print(f"Running parallel search, got {total_correct_parallel / (k * num_queries)} recall on {num_queries} queries,"
          f" and {num_queries / total_search_time_parallel} query per seconds")
    print(f"Got {total_search_time / total_search_time_parallel} times improvement in runtime using {n_threads} threads\n")

    # Validate that the recall of the parallel search recall is the same as the sequential search recall.
    assert total_correct_parallel == total_correct


def test_parallel_insert():
    k = 10
    num_queries = 10000
    n_threads = min(os.cpu_count(), 8)
    expected_parallel_rate = 0.9  # we expect that at least 90% of the insert/search time will be executed in parallel

    print(f"Inserting {num_elements} vectors of dim {dim} into HNSW sequentially took"
          f" {g_test_index.sequential_insert_time} seconds")

    parallel_index = create_hnsw_index(g_test_index.dim, g_test_index.num_elements, g_test_index.metric,
                                       g_test_index.data_type, ef_runtime=200)
    start = time.time()
    parallel_index.add_vector_parallel(g_test_index.data, np.array(range(num_elements)), n_threads)
    parallel_insert_time = time.time() - start
    assert parallel_index.index_size() == num_elements
    assert parallel_index.check_integrity()
    # Validate that the parallel index contains the same vectors as the sequential one.
    for label in range(num_elements):
        assert_allclose(g_test_index.hnsw_index.get_vector(label), parallel_index.get_vector(label))

    print(f"Inserting {num_elements} vectors of dim {dim} into HNSW in parallel took {parallel_insert_time} seconds")
    print(f"Got {g_test_index.sequential_insert_time/parallel_insert_time} times improvement using {n_threads} threads\n")

    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index.compute_ground_truth_knn(query_data, k)

    # Run search over the baseline hnsw index (that was created by inserting vectors one by one).
    start = time.time()
    res_labels, _ = g_test_index.hnsw_index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time = time.time() - start

    total_correct = 0
    for i in range(num_queries):
        total_correct += len(set(g_test_index.total_res_bf[i]).intersection(set(res_labels[i])))
    print(f"Running parallel search over an index that was built by inserting vectors one by one, got"
          f" {total_correct / (k * num_queries)} recall on {num_queries} queries,"
          f" with {num_queries/total_search_time} query per second")

    # Run search with parallel index and assert that similar recall achieved.
    start = time.time()
    res_labels, _ = parallel_index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time_parallel = time.time() - start

    total_correct_parallel = 0
    for i in range(num_queries):
        total_correct_parallel += len(set(g_test_index.total_res_bf[i]).intersection(set(res_labels[i])))
    print(f"Running parallel search on index that was created using parallel insert, got "
          f"{total_correct_parallel / (k * num_queries)} recall on {num_queries} queries, and"
          f" {num_queries/total_search_time_parallel} query per second")
    assert total_correct_parallel >= total_correct * 0.95  # 0.95 is an arbitrary threshold


def test_parallel_insert_search():
    k = 10
    num_queries = 10000
    n_threads = min(os.cpu_count(), 8)

    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index.compute_ground_truth_knn(query_data, k)

    # Insert vectors to the index and search in parallel.
    parallel_index = create_hnsw_index(g_test_index.dim, g_test_index.num_elements, g_test_index.metric,
                                       g_test_index.data_type, ef_runtime=200)

    def insert_vectors():
        parallel_index.add_vector_parallel(g_test_index.data, np.array(range(num_elements)), num_threads=int(n_threads/2))

    res_labels_g = np.zeros((num_queries, dim))

    def run_queries():
        nonlocal res_labels_g
        res_labels_g, _ = parallel_index.knn_parallel(query_data, k, num_threads=int(n_threads/2))

    t_insert = threading.Thread(target=insert_vectors)
    t_query = threading.Thread(target=run_queries)
    print("Running KNN queries in parallel to inserting vectors to the index, start running queries after more 50% of"
          " the vectors are indexed")
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
            total_correct_cur_chunk += len(set(g_test_index.total_res_bf[j]).intersection(set(res_labels_g[j])))
        assert total_correct_cur_chunk >= total_correct_prev_chunk
        total_correct_prev_chunk = total_correct_cur_chunk
        print(f"Recall for chunk {int(i/chunk_size)+1}/{int(num_queries/chunk_size)} of queries is:"
              f" {total_correct_cur_chunk/(k*chunk_size)}")


def test_parallel_with_range():
    num_queries = 10000
    radius = 3.0
    n_threads = min(os.cpu_count(), 8)
    PADDING_LABEL = -1  # used for padding empty labels entries in a single query results
    expected_parallel_rate = 0.9  # we expect that at least 90% of the insert/search time will be executed in parallel

    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index.compute_ground_truth_range(query_data, radius)

    # Run serial range queries
    total_search_time = 0
    # The ratio between then number of results returned by HNSW and the total number of vectors in the range.
    overall_intersection_rate = 0

    total_results = 0
    for i, query in enumerate(query_data):
        start = time.time()
        res_labels_range, res_distances_range = g_test_index.hnsw_index.range_query(query, radius)
        total_search_time += time.time() - start
        assert set(res_labels_range[0]).issubset(set(g_test_index.total_res_bf[i]))
        total_results += g_test_index.total_res_bf[i].size
        overall_intersection_rate += res_labels_range[0].size / g_test_index.total_res_bf[i].size \
            if g_test_index.total_res_bf[i].size > 0 else 1
    print(f"Range queries - running {num_queries} queries sequentially, average number of results is:"
          f" {total_results/num_queries} and HNSW success rate is: {overall_intersection_rate/num_queries}."
          f" query per seconds: {num_queries/total_search_time}")

    # Run range queries in parallel
    start = time.time()
    hnsw_labels_range_parallel, _ = g_test_index.hnsw_index.range_parallel(query_data, radius=radius)
    total_range_query_parallel_time = time.time() - start
    overall_intersection_rate_parallel = 0
    for i in range(num_queries):
        query_results_set = set(hnsw_labels_range_parallel[i])
        query_results_set.discard(PADDING_LABEL)  # remove the irrelevant padding values
        assert query_results_set.issubset(set(g_test_index.total_res_bf[i]))
        overall_intersection_rate_parallel += len(query_results_set) / g_test_index.total_res_bf[i].size \
            if g_test_index.total_res_bf[i].size > 0 else 1
    print(f"Running the same {num_queries} queries in parallel, query per seconds is"
          f" {num_queries/total_range_query_parallel_time}, and intersection rate is: "
          f"{overall_intersection_rate_parallel/num_queries}")
    assert overall_intersection_rate_parallel == overall_intersection_rate
    print(f"Got improvement of {total_search_time/total_range_query_parallel_time} times using {n_threads} threads\n")


def test_parallel_insert_multi():
    k = 10
    num_labels = int(g_test_index_multi.num_elements / g_test_index_multi.vectors_per_label)
    num_queries = 10000
    n_threads = min(os.cpu_count(), 8)
    expected_parallel_rate = 0.85  # we expect that at least 85% of the insert/search time will be executed in parallel

    print(f"Inserting {num_elements} vectors of dim {dim} into multi-HNSW ({per_label} vectors per label) sequentially"
          f" took {g_test_index_multi.sequential_insert_time} seconds")

    parallel_multi_index = create_hnsw_index(g_test_index_multi.dim, g_test_index_multi.num_elements,
                                             g_test_index_multi.metric, g_test_index_multi.data_type, ef_runtime=200,
                                             is_multi=True)

    # Insert vectors to multi index in parallel
    data = g_test_index_multi.data.reshape(num_elements, dim)
    labels = np.concatenate([[i]*g_test_index_multi.vectors_per_label for i in range(num_labels)])
    start = time.time()
    parallel_multi_index.add_vector_parallel(data, labels, n_threads)
    parallel_insert_time = time.time() - start
    assert parallel_multi_index.index_size() == num_elements
    assert parallel_multi_index.check_integrity()
    # Validate that the parallel index contains the same vectors as the sequential one. vectors are not necessarily
    # at the same order, so we flatten the array and check that elements are set equal.
    for label in range(num_labels):
        vectors_s = g_test_index_multi.hnsw_index.get_vector(label)
        vectors_p = parallel_multi_index.get_vector(label)
        assert vectors_s.shape == vectors_p.shape
        assert set(vectors_s.flatten()) == set(vectors_p.flatten())

    print(f"Inserting {num_elements} vectors of dim {dim} into multi-HNSW in parallel ({per_label} vectors per label)"
          f" took {parallel_insert_time} seconds")
    print(f"Got {g_test_index_multi.sequential_insert_time/parallel_insert_time} times improvement using {n_threads} threads\n")

    # Run queries over the multi-index
    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index_multi.compute_ground_truth_knn(query_data, k)

    # Run search over the baseline hnsw index (that was created by inserting vectors one by one).
    total_search_time = 0
    total_correct = 0
    for i, query in enumerate(query_data):
        start = time.time()
        res_labels, _ = g_test_index_multi.hnsw_index.knn_query(query, k)
        total_search_time += time.time() - start
        total_correct += len(set(res_labels[0]).intersection(g_test_index_multi.total_res_bf[i]))
    print(f"Running search over baseline multi index, got {total_correct / (k * num_queries)} recall on {num_queries}"
          f" queries, and {num_queries/total_search_time} query per second")

    # Run search with parallel index and assert that similar recall achieved.
    start = time.time()
    res_labels_parallel, res_dists_parallel = parallel_multi_index.knn_parallel(query_data, k, num_threads=n_threads)
    total_search_time_parallel = time.time() - start
    total_correct_parallel = 0
    for res_labels, ground_truth in zip(res_labels_parallel, g_test_index_multi.total_res_bf):
        total_correct_parallel += len(set(res_labels).intersection(set(ground_truth)))

    print(f"Running parallel search over multi index built in parallel, got {total_correct_parallel / (k * num_queries)}"
          f" recall on {num_queries} queries, and {num_queries/total_search_time_parallel} query per second")
    print(f"Got {total_search_time / total_search_time_parallel} times improvement in runtime using"
          f" {n_threads} threads\n")
    assert total_correct_parallel >= total_correct * 0.95  # 0.95 is an arbitrary threshold


def test_parallel_multi_insert_search():
    k = 10
    num_queries = 10000
    n_threads = min(os.cpu_count(), 8)
    num_labels = int(g_test_index_multi.num_elements / g_test_index_multi.vectors_per_label)

    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index_multi.compute_ground_truth_knn(query_data, k)

    # Insert vectors to the index and search in parallel.
    parallel_multi_index = create_hnsw_index(g_test_index_multi.dim, g_test_index_multi.num_elements,
                                             g_test_index_multi.metric, g_test_index_multi.data_type, ef_runtime=200,
                                             is_multi=True)

    data = g_test_index_multi.data.reshape(num_elements, dim)
    labels = np.concatenate([[i]*g_test_index_multi.vectors_per_label for i in range(num_labels)])

    def insert_vectors():
        parallel_multi_index.add_vector_parallel(data, labels, num_threads=int(n_threads/2))

    res_labels_g = np.zeros((num_queries, dim))

    def run_queries():
        nonlocal res_labels_g
        res_labels_g, _ = parallel_multi_index.knn_parallel(query_data, k, num_threads=int(n_threads/2))

    t_insert = threading.Thread(target=insert_vectors)
    t_query = threading.Thread(target=run_queries)
    print("Running KNN queries in parallel to inserting vectors to the multi index, start running queries after more"
          " 50% of the vectors are indexed")
    t_insert.start()
    # Wait until half of the index is indexed, then start run queries
    while parallel_multi_index.index_size() < num_elements / 2:
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
            total_correct_cur_chunk += len(set(g_test_index_multi.total_res_bf[j]).intersection(set(res_labels_g[j])))
        assert total_correct_cur_chunk >= total_correct_prev_chunk
        total_correct_prev_chunk = total_correct_cur_chunk
        print(f"Recall for queries' chunk {int(i/chunk_size)+1}/{int(num_queries/chunk_size)} is:"
              f" {total_correct_cur_chunk/(k*chunk_size)}")


def test_parallel_batch_search():
    num_queries = 10000
    batch_size = 100
    n_batches = 5
    n_threads = min(os.cpu_count(), 8)
    expected_parallel_rate = 0.85  # we expect that at least 85% of the insert/search time will be executed in parallel

    # Sequential batched search as the baseline
    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index.compute_ground_truth_knn(query_data, batch_size*n_batches)
    total_search_time = 0
    total_correct = 0
    for i, query in enumerate(query_data):
        start = time.time()
        batch_iterator = g_test_index.hnsw_index.create_batch_iterator(query)
        # Collect all the results from all batches
        res_labels = set()
        for _ in range(n_batches):
            res_labels = res_labels.union(set(batch_iterator.get_next_results(batch_size, BY_SCORE)[0][0]))

        total_search_time += time.time() - start
        total_correct += len(res_labels.intersection(set(g_test_index.total_res_bf[i])))

    print(f"Running sequential batched search of {n_batches} batches of size {batch_size}, over {num_queries} queries,"
          f" got recall of {total_correct/(n_batches*batch_size*num_queries)} and "
          f" {num_queries/total_search_time} query per second")

    total_results_parallel = {}

    def run_batched_search(query_, query_ind):
        batch_iterator_ = g_test_index.hnsw_index.create_batch_iterator(query_)
        res_labels_ = set()
        for _ in range(n_batches):
            res_labels_ = res_labels_.union(set(batch_iterator_.get_next_results(batch_size, BY_SCORE)[0][0]))
        total_results_parallel[query_ind] = res_labels_

    start = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(run_batched_search, q, i) for i, q in enumerate(query_data)]
        done, not_done = wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    total_search_time_parallel = time.time() - start
    assert len(done) == num_queries and len(not_done) == 0

    total_correct_parallel = 0
    for i in range(num_queries):
        total_correct_parallel += len(set(g_test_index.total_res_bf[i]).intersection(total_results_parallel[i]))

    print(f"Running parallel batched search of {n_batches} batches of size {batch_size}, over {num_queries} queries,"
          f" got recall of {total_correct_parallel/(n_batches*batch_size*num_queries)} and "
          f" {num_queries/total_search_time_parallel} query per second")
    print(f"Got {total_search_time / total_search_time_parallel} times improvement in runtime using "
          f"{n_threads} threads\n")

    # Validate that the recall of the parallel search recall is the same as the sequential search recall.
    assert total_correct_parallel == total_correct


def test_parallel_insert_batch_search():
    num_queries = 10000
    batch_size = 100
    n_batches = 5
    n_threads = min(os.cpu_count(), 8)

    # Insert vectors to the index and search in parallel.
    parallel_index = create_hnsw_index(g_test_index.dim, g_test_index.num_elements, g_test_index.metric,
                                       g_test_index.data_type, ef_runtime=200)

    query_data = np.float32(np.random.random((num_queries, dim)))
    g_test_index.compute_ground_truth_knn(query_data, n_batches*batch_size)

    total_results_parallel = {}

    def run_batched_search(query_, query_ind):
        nonlocal total_results_parallel
        batch_iterator_ = parallel_index.create_batch_iterator(query_)
        res_labels_ = set()
        for _ in range(n_batches):
            res_labels_ = res_labels_.union(set(batch_iterator_.get_next_results(batch_size, BY_SCORE)[0][0]))
        total_results_parallel[query_ind] = res_labels_

    def insert_vectors():
        parallel_index.add_vector_parallel(g_test_index.data, range(num_elements), num_threads=int(n_threads/2))

    t_insert = threading.Thread(target=insert_vectors)
    print("Running batched search in parallel to inserting vectors to the index, start running queries after more 50%"
          " of the vectors are indexed")
    t_insert.start()
    # Wait until half of the index is indexed, then start run queries
    while parallel_index.index_size() < num_elements / 2:
        time.sleep(0.5)

    with ThreadPoolExecutor(max_workers=int(n_threads/2)) as executor:
        futures = [executor.submit(run_batched_search, q, i) for i, q in enumerate(query_data)]
        done, not_done = wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    assert len(done) == num_queries and len(not_done) == 0

    t_insert.join()
    assert parallel_index.index_size() == num_elements
    assert parallel_index.check_integrity()

    # Measure recall - expect to get increased recall over time, since vectors are being inserted while queries
    # are running, and the ground truth is measured compared to the index that contains all the elements.
    chunk_size = int(num_queries/5)
    total_correct_prev_chunk = 0
    for i in range(0, num_queries, chunk_size):
        total_correct_cur_chunk = 0
        for j in range(i, i+chunk_size):
            total_correct_cur_chunk += len(set(g_test_index.total_res_bf[j]).intersection(total_results_parallel[j]))
        assert total_correct_cur_chunk >= total_correct_prev_chunk
        total_correct_prev_chunk = total_correct_cur_chunk
        print(f"Recall for chunk {int(i/chunk_size)+1}/{int(num_queries/chunk_size)} of queries is:"
              f" {total_correct_cur_chunk/(batch_size*n_batches*chunk_size)}")
