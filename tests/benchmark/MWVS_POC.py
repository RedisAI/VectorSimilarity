import heapq
import numpy as np
import time
from VecSim import *
import h5py
from functools import reduce


class MultiWeightedTopKQuery:
    def __init__(self, indexes, queries, weights, k):
        self.indexes = indexes
        self.queries = queries
        self.weights = weights
        self.k = k
        self.bf_results = None
        self.bf_computation_time = None
        self.combined_knn_top_results = None
        self.combined_knn_computation_time = None
        self.combined_knn_top_results = None
        self.combined_knn_computation_time = None
        self.threshold = None
        self.combined_range_query_top_results = None
        self.combined_range_query_computation_time = None

    def compute_bf_results(self):
        start = time.time()
        h = []  # min-heap (priority queue), use -1*score as the priority
        for i in range(self.indexes[0].index_size()):
            combined_score = 0
            for j in range(len(self.indexes)):
                combined_score += self.weights[j] * self.indexes[j].get_distance_from(self.queries[j], i)
            if len(h) < self.k:
                heapq.heappush(h, (-combined_score, i))
            elif -h[0][0] > combined_score:
                heapq.heapreplace(h, (-combined_score, i))
        end = time.time()
        # for e in h:
        #     print(f"combined score of {e[1]} is {-e[0]}")
        #     print(f"dbpedia score is {self.indexes[0].get_distance_from(self.queries[0], e[1])}")
        #     print(f"glove-50 score is {self.indexes[1].get_distance_from(self.queries[1], e[1])}")
        #     print(f"glove-200 score is {self.indexes[2].get_distance_from(self.queries[2], e[1])}")
        self.bf_results = h
        self.bf_computation_time = end-start

    def compute_recall(self, hnsw_res_ids, time_field):
        combined_scores = []
        start = time.time()
        for res in hnsw_res_ids:
            combined_score = 0
            for j in range(len(self.indexes)):
                combined_score += self.weights[j] * self.indexes[j].get_distance_from(self.queries[j], res)
            combined_scores.append(combined_score)
        end = time.time()
        time_field += end-start

        # measure recall of the HNSW results comparing to the BF results
        actual_res_ids = [res[1] for res in self.bf_results]
        correct = 0
        for res in hnsw_res_ids:
            for actual_res in actual_res_ids:
                if res == actual_res:
                    correct += 1
                    break

        return correct/self.k, combined_scores


# Load serialized HNSW index from file.
def create_index(params, index_file_name, use_flat=False, data_set=None):
    if use_flat:
        bf_params = BFParams()
        bf_params.initialCapacity = params.initialCapacity
        bf_params.dim = params.dim
        bf_params.type = VecSimType_FLOAT32
        bf_params.metric = VecSimMetric_Cosine
        bf_index = BFIndex(bf_params)

        for i, vector in enumerate(data_set):
            if i == 1000000:
                break
            bf_index.add_vector(vector, i)

        print(f"Created FLAT index from {index_file_name} data, index size is {bf_index.index_size()}.")
        return bf_index

    hnsw_index = HNSWIndex(params)
    hnsw_index.load_index(index_file_name)

    print(f"Loaded HNSW index from {index_file_name}, index size is {hnsw_index.index_size()}.")
    return hnsw_index


def load_index_and_test_set(dateset):
    file_name = "/home/alon/Code/VectorSimilarity/tests/benchmark/data/"+dateset
    if dateset != "dbpedia-768":
        file_name += "-angular"
    file_name += ".hdf5"
    return np.array(h5py.File(file_name, 'r')['train']), np.array(h5py.File(file_name, 'r')['test'])


def get_combined_top_k_results(query, r):
    start = time.time()
    total_res_ids = []
    for i in range(len(query.indexes)):
        res_index_ids, res_index_scores = query.indexes[i].knn_query(query.queries[i], r*query.k)
        total_res_ids.append(res_index_ids[0])
    total_res_ids = reduce(np.union1d, total_res_ids).tolist()
    end = time.time()
    query.combined_knn_computation_time = end-start

    recall, combined_scores = query.compute_recall(total_res_ids, query.combined_knn_computation_time)

    query.combined_knn_top_results = total_res_ids
    query.threshold = sorted(combined_scores)[query.k - 1]  # threshold for the upcoming range query
    # print(f"threshold is {query.threshold}")

    return recall


def get_combined_range_results(query, r, epsilon=0.01):

    query_params = None
    if epsilon is not None:
        query_params = VecSimQueryParams()
        query_params.hnswRuntimeParams.epsilon = epsilon

    start = time.time()
    total_res_ids = [np.array(query.combined_knn_top_results, dtype=np.uint32)]
    # total_res_ids = []
    for i in range(len(query.indexes)):
        res_index_ids, res_index_scores = query.indexes[i].range_query(query.queries[i], r*query.threshold, query_params)
        total_res_ids.append(res_index_ids[0])
    total_res_ids = reduce(np.union1d, total_res_ids).tolist()
    end = time.time()

    query.combined_range_query_top_results = total_res_ids
    query.combined_range_query_computation_time = end-start
    recall, _ = query.compute_recall(total_res_ids, query.combined_range_query_computation_time)

    return recall


def setup(data_sets, use_flat=False):
    test_sets = []
    dataset_size = 1000000

    # DBPedia #
    hnswparams = HNSWParams()
    hnswparams.M = 64
    hnswparams.efConstruction = 512
    hnswparams.initialCapacity = dataset_size
    hnswparams.efRuntime = 10
    hnswparams.dim = 768
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine

    data_set, test_set = load_index_and_test_set(data_sets[0])
    dbpedia_index = create_index(hnswparams,
                                 "/home/alon/Code/VectorSimilarity/tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1")
    test_sets.append(test_set)

    # Glove-50 #
    hnswparams = HNSWParams()
    hnswparams.M = 24
    hnswparams.efConstruction = 150
    hnswparams.initialCapacity = dataset_size
    hnswparams.efRuntime = 200
    hnswparams.dim = 50
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine

    data_set, test_set = load_index_and_test_set(data_sets[1])
    glove_50_index = create_index(hnswparams, "/home/alon/Code/VectorSimilarity/tests/benchmark/data/glove-50-angular-M=24-ef=150-trimmed_to_1M.hnsw",
                                  use_flat=use_flat, data_set=data_set)
    # glove_50_index.save_index("tests/benchmark/data/glove-50-angular-M=24-ef=150-trimmed_to_1M.hnsw")
    test_sets.append(test_set)

    # Glove-200 #
    hnswparams = HNSWParams()
    hnswparams.M = 48
    hnswparams.efConstruction = 350
    hnswparams.initialCapacity = dataset_size
    hnswparams.efRuntime = 350
    hnswparams.dim = 200
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine

    data_set, test_set = load_index_and_test_set(data_sets[2])
    glove_200_index = create_index(hnswparams, "/home/alon/Code/VectorSimilarity/tests/benchmark/data/glove-200-angular-M=48-ef=350-trimmed_to_1M.hnsw")
    # glove_200_index.save_index("tests/benchmark/data/glove-200-angular-M=48-ef=350-trimmed_to_1M.hnsw")
    test_sets.append(test_set)

    print("3 Indexes loaded successfully\n")
    return [dbpedia_index, glove_50_index, glove_200_index], test_sets


def prepare_queries(indexes, weights, test_sets, num_queries, k):
    print(f"Running {num_queries} queries with k={k}, weights are: {weights}")
    print("Computing BF results...")
    queries = []
    total_time = 0
    for i in range(num_queries):
        q0 = test_sets[0][i]
        normalized_q0 = q0 / np.sqrt(np.sum(q0**2))
        q1 = test_sets[1][i]
        normalized_q1 = q1 / np.sqrt(np.sum(q1**2))
        q2 = test_sets[2][i]
        normalized_q2 = q2 / np.sqrt(np.sum(q2**2))

        query = MultiWeightedTopKQuery(indexes, [normalized_q0, normalized_q1, normalized_q2], weights, k)
        query.compute_bf_results()
        total_time += query.bf_computation_time
        queries.append(query)
    print(f"Computing BF results took an average time of {total_time/num_queries} per query")
    return queries


def run_standard_knn(queries):
    n_queries = len(queries)

    # Compute results for combined knn search only, for several values of r_knn
    for r_knn in [1, 10, 100]:
        print(f"\n***Running knn - asking for {r_knn}*k results for every individual knn query***")
        total_recall = 0
        total_time = 0
        for query in queries:
            recall = get_combined_top_k_results(query, r_knn)
            total_time += query.combined_knn_computation_time
            total_recall += recall

        print(f"Computing with only knn took an average time of {total_time/n_queries} per query,"
              f" with avg. {total_recall/n_queries} recall")

        # Compute results for combined range query, where the range derive from the previous phase.
        for r_range in [0.7, 0.8, 0.9, 1]:
            print(f"\nRunning second phase: range query with radius={r_range}*threshold,"
                  f" where the threshold is the combined score of the k-th result (from the previous phase)")
            total_recall = 0
            total_time = 0
            total_res = 0
            for query in queries:
                recall = get_combined_range_results(query, r_range)
                total_recall += recall
                total_time += query.combined_range_query_computation_time + query.combined_knn_computation_time
                total_res += len(query.combined_range_query_top_results)

            print(f"Computing with range query took an average time of {total_time/n_queries} per query,"
                  f" with avg. res of {total_res/n_queries} and {total_recall/n_queries} recall")


def main():
    # Create 2 lists of three vector indexes and their corresponding vectors test sets. By default, all three
    # indexes are HNSW, optionally one (glove-50) is flat.
    data_sets = ["dbpedia-768", "glove-50", "glove-200"]
    vector_indexes, test_sets = setup(data_sets, use_flat=False)

    # Run multi top k search over queries from the test set
    k = 10
    num_queries = 1
    w0 = 1/3
    w1 = 1/3
    w2 = 1/3
    queries = prepare_queries(vector_indexes, [w0, w1, w2], test_sets, num_queries, k)
    run_standard_knn(queries)


if __name__ == '__main__':
    main()
