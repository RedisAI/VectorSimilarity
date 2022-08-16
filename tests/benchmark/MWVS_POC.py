import heapq
import os
from urllib.request import urlretrieve
import numpy as np
import time
from VecSim import *
import h5py
from functools import reduce


# Load serialized HNSW index from file.
def create_index(params, index_file_name, use_BF=False, data_set=None):
    if use_BF:
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
    file_name = "tests/benchmark/data/"+dateset
    if dateset != "dbpedia-768":
        file_name += "-angular"
    file_name += ".hdf5"
    return np.array(h5py.File(file_name, 'r')['train']), np.array(h5py.File(file_name, 'r')['test'])


def get_BF_results(dbpedia_index, glove_50_index, glove_200_index, q0, q1, q2, w0, w1, w2, k):
    start = time.time()
    h = []  # min-heap (priority queue), use -1*score as the priority
    for i in range(dbpedia_index.index_size()):
        combined_score = w0 * dbpedia_index.get_distance_from(q0, i) + \
                         w1 * glove_50_index.get_distance_from(q1, i) + \
                         w2 * glove_200_index.get_distance_from(q2, i)
        if len(h) < k:
            heapq.heappush(h, (-combined_score, i))
        elif -h[0][0] > combined_score:
            heapq.heapreplace(h, (-combined_score, i))
    end = time.time()
    # for e in h:
    #     print(f"combined score of {e[1]} is {-e[0]}")
    #     print(f"dbpedia score is {dbpedia_index_flat.get_distance_from(q0, e[1])}")
    #     print(f"glove-50 score is {glove_50_index_flat.get_distance_from(q1, e[1])}")
    #     print(f"glove-200 score is {glove_200_index_flat.get_distance_from(q2, e[1])}")

    return h, end-start


def compute_recall(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                   normalized_q1, normalized_q2, w0, w1, w2, k, HNSW_res_ids, BF_res):

    combined_scores = []
    for i, res in enumerate(HNSW_res_ids):
        combined_score = w0 * dbpedia_index.get_distance_from(normalized_q0, HNSW_res_ids[i]) + \
                         w1 * glove_50_index.get_distance_from(normalized_q1, HNSW_res_ids[i]) + \
                         w2 * glove_200_index.get_distance_from(normalized_q2, HNSW_res_ids[i])
        combined_scores.append(combined_score)

    # measure recall of the HNSW results comparing to the BF results
    actual_res_ids = [res[1] for res in BF_res]
    correct = 0
    for res in HNSW_res_ids:
        for actual_res in actual_res_ids:
            if res == actual_res:
                correct += 1
                break

    return correct/k, combined_scores


def get_combined_top_k_results(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                                 normalized_q1, normalized_q2, w0, w1, w2, k, r, BF_res):
    start = time.time()
    res_dbpedia_ids, res_dbpedia_scores = dbpedia_index.knn_query(normalized_q0, r*k)
    res_glove_50_ids, res_glove_50_scores = glove_50_index.knn_query(normalized_q1, r*k)
    res_glove_200_ids, res_glove_50_scores = glove_200_index.knn_query(normalized_q2, r*k)
    total_res = reduce(np.union1d, (res_dbpedia_ids[0], res_glove_50_ids[0], res_glove_200_ids[0])).tolist()
    end = time.time()

    recall, combined_scores = compute_recall(dbpedia_index, glove_50_index, glove_200_index, normalized_q0, normalized_q1, normalized_q2,
                            w0, w1, w2, k, total_res, BF_res)
    # print(f"Current recall is {recall} from {len(total_res)} results obtained in top k search, total search time took {end-start}")

    T = sorted(combined_scores)[k-1]  # threshold for the upcoming range query
    return recall, end-start, T
    # return correct/k, end-start


def get_combined_range_results(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                               normalized_q1, normalized_q2, w0, w1, w2, k, r, threshold, epsilon, BF_res):

    query_params = VecSimQueryParams()
    query_params.hnswRuntimeParams.epsilon = epsilon
    start = time.time()
    res_dbpedia_ids, res_dbpedia_scores = dbpedia_index.range_query(normalized_q0, threshold*r, query_params)
    res_glove_50_ids, res_glove_50_scores = glove_50_index.range_query(normalized_q1, threshold*r, query_params)
    res_glove_200_ids, res_glove_200_scores = glove_200_index.range_query(normalized_q2, threshold*r, query_params)
    total_res = reduce(np.union1d, (res_dbpedia_ids[0], res_glove_50_ids[0], res_glove_200_ids[0])).tolist()
    end = time.time()

    recall, combined_scores = compute_recall(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                                             normalized_q1, normalized_q2, w0, w1, w2, k, total_res, BF_res)

    # print(f"Current recall is {recall} from {len(total_res)} results obtained in range search, total search time took {end-start}")
    return recall, end-start, len(total_res)


def main():
    # Create 3 HNSW indexes and tests_sets:
    test_sets = {}
    data_sets = ["dbpedia-768", "glove-50", "glove-200"]
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
    dbpedia_index = create_index(hnswparams, "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1")
    test_sets[data_sets[0]] = test_set

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
    glove_50_index = create_index(hnswparams, "tests/benchmark/data/glove-50-angular-M=24-ef=150-trimmed_to_1M.hnsw",
                                  use_BF=True, data_set=data_set)
    # glove_50_index.save_index("tests/benchmark/data/glove-50-angular-M=24-ef=150-trimmed_to_1M.hnsw")
    test_sets[data_sets[1]] = test_set

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
    glove_200_index = create_index(hnswparams, "tests/benchmark/data/glove-200-angular-M=48-ef=350-trimmed_to_1M.hnsw")
    # glove_200_index.save_index("tests/benchmark/data/glove-200-angular-M=48-ef=350-trimmed_to_1M.hnsw")
    test_sets[data_sets[2]] = test_set

    print("3 Indexes loaded successfully\n")

    # Run multi top k search (standard KNN)
    k = 10
    num_queries = 100
    w0 = w1 = 1/3
    w2 = 1/3
    print(f"Running {num_queries} queries with k={k}, w0={w0}, w1={w1}, w2={w2}...\n")

    # Collect the BF results of the test queries.
    BF_res_lookup = []
    print("Computing BF results...")
    total_time = 0
    for i in range(num_queries):
        q0 = test_sets[data_sets[0]][i]
        normalized_q0 = q0 / np.sqrt(np.sum(q0**2))
        q1 = test_sets[data_sets[1]][i]
        normalized_q1 = q1 / np.sqrt(np.sum(q1**2))
        q2 = test_sets[data_sets[2]][i]
        normalized_q2 = q2 / np.sqrt(np.sum(q2**2))

        BF_res, time = get_BF_results(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                            normalized_q1, normalized_q2, w0, w1, w2, k)
        BF_res_lookup.append(BF_res)
        total_time += time
    print(f"Computing BF results took an average time of {total_time/num_queries} per query")

    # Compute with HNSW knn/range and compute recall.
    r_knn = 10
    # r_range = 1
    for r_range in [0.6, 0.75, 0.9]:
        print(f"range is {r_range} out of the initial k-th combined score (in the KNN phase)")
        # print(f"Running knn with k={r_knn}*k for every individual knn query")
        total_recall = 0
        total_time = 0
        total_res = 0
        for i in range(num_queries):
            # print(f"query no. {i}: ")
            q0 = test_sets[data_sets[0]][i]
            normalized_q0 = q0 / np.sqrt(np.sum(q0**2))
            q1 = test_sets[data_sets[1]][i]
            normalized_q1 = q1 / np.sqrt(np.sum(q1**2))
            q2 = test_sets[data_sets[2]][i]
            normalized_q2 = q2 / np.sqrt(np.sum(q2**2))

            recall, top_k_time, threshold = get_combined_top_k_results(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                                                        normalized_q1, normalized_q2, w0, w1, w2, k, r_knn, BF_res_lookup[i])

            # print(f"Threshold is {threshold}")
            recall, range_time, res = get_combined_range_results(dbpedia_index, glove_50_index, glove_200_index, normalized_q0,
                normalized_q1, normalized_q2, w0, w1, w2, k, r_range, threshold, 0.01, BF_res_lookup[i])
            total_recall += recall
            # total_time += top_k_time
            total_time += top_k_time+range_time
            total_res += res
        print(f"Computing with range query took an average time of {total_time/num_queries} per query,"
              f" with avg. res of {total_res/num_queries} and {total_recall/num_queries} recall")
        # print(f"Computing with only knn took an average time of {total_time/num_queries} per query,"
        #       f" with avg. {total_recall/num_queries} recall")


if __name__ == '__main__':
    main()
