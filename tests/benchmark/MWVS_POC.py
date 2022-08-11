import heapq
import os
from urllib.request import urlretrieve
import numpy as np
import time
from VecSim import *
import h5py
from functools import reduce


# Load serialized HNSW index from file, and build a corresponding BF index
def create_indexes(params, data_set, index_file_name):
    hnsw_index = HNSWIndex(params)
    hnsw_index.load_index(index_file_name)

    bfparams = BFParams()
    bfparams.initialCapacity = params.initialCapacity
    bfparams.dim = params.dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_Cosine
    flat_index = BFIndex(bfparams)
    for i, vector in enumerate(data_set[:1000000]):
        flat_index.add_vector(vector, i)

    index_size = hnsw_index.index_size()
    for i in range(1000000, index_size):
        hnsw_index.delete_vector(i)

    print(f"loaded HNSW index from file {index_file_name}, index of size {hnsw_index.index_size()}, "
          f"and created a corresponding flat index of size {flat_index.index_size()}")
    return hnsw_index, flat_index


def load_index_and_test_set(dateset):
    file_name = "tests/benchmark/data/"+dateset
    if dateset != "dbpedia-768":
        file_name += "-angular"
    file_name += ".hdf5"
    return np.array(h5py.File(file_name, 'r')['train']), np.array(h5py.File(file_name, 'r')['test'])


def get_BF_results(dbpedia_index_flat, glove_50_index_flat, glove_200_index_flat, q0, q1, q2, w0, w1, w2, k):
    h = []  # min-heap (priority queue), use -1*score as the priority
    for i in range(dbpedia_index_flat.index_size()):
        combined_score = w0 * dbpedia_index_flat.get_distance_from(q0, i) + \
                         w1 * glove_50_index_flat.get_distance_from(q1, i) + \
                         w2 * glove_200_index_flat.get_distance_from(q2, i)
        if len(h) < k:
            heapq.heappush(h, (-combined_score, i))
        elif -h[0][0] > combined_score:
            heapq.heapreplace(h, (-combined_score, i))

    # for e in h:
    #     print(f"combined score of {e[1]} is {-e[0]}")
    #     print(f"dbpedia score is {dbpedia_index_flat.get_distance_from(q0, e[1])}")
    #     print(f"glove-50 score is {glove_50_index_flat.get_distance_from(q1, e[1])}")
    #     print(f"glove-200 score is {glove_200_index_flat.get_distance_from(q2, e[1])}")

    return h


def main():
    # Create 3 HNSW indexes and tests_sets:
    test_sets = {}
    data_sets = ["dbpedia-768", "glove-50", "glove-200"]
    dataset_size = 1000000

    # DBPedia #
    hnswparams = HNSWParams()
    hnswparams.M = 64
    hnswparams.efConstruction = 512
    hnswparams.initialCapacity = 1000000
    hnswparams.efRuntime = 10
    hnswparams.dim = 768
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine

    data_set, test_set = load_index_and_test_set(data_sets[0])
    dbpedia_index, dbpedia_index_flat =\
        create_indexes(hnswparams, data_set, "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1")
    test_sets[data_sets[0]] = test_set

    # Glove-50 #
    hnswparams = HNSWParams()
    hnswparams.M = 24
    hnswparams.efConstruction = 150
    hnswparams.initialCapacity = 1183514
    hnswparams.efRuntime = 200
    hnswparams.dim = 50
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine

    data_set, test_set = load_index_and_test_set(data_sets[1])
    glove_50_index, glove_50_index_flat =\
        create_indexes(hnswparams, data_set, "tests/benchmark/data/glove-50-angular-M=24-ef=150.hnsw")
    test_sets[data_sets[1]] = test_set

    # Glove-200 #
    hnswparams = HNSWParams()
    hnswparams.M = 48
    hnswparams.efConstruction = 350
    hnswparams.initialCapacity = 1183514
    hnswparams.efRuntime = 350
    hnswparams.dim = 200
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine

    data_set, test_set = load_index_and_test_set(data_sets[2])
    glove_200_index, glove_200_index_flat =\
        create_indexes(hnswparams, data_set, "tests/benchmark/data/glove-200-angular-M=48-ef=350.hnsw")
    test_sets[data_sets[2]] = test_set

    print("3 Indexes loaded successfully\n")

    # Run multi top k search (standard KNN)
    print("Running queries...")
    k = 10
    w0 = w1 = w2 = 1/3
    for i in range(5):
        q0 = test_sets[data_sets[0]][i]
        q1 = test_sets[data_sets[1]][i]
        q2 = test_sets[data_sets[2]][i]

        BF_res = get_BF_results(dbpedia_index_flat, glove_50_index_flat, glove_200_index_flat,
                                                   q0, q1, q2, w0, w1, w2, k)

        #print(BF_res)
        res_dbpedia_ids, res_dbpedia_scores = dbpedia_index.knn_query(q0, k)
        res_glove_50_ids, res_glove_50_scores = glove_50_index.knn_query(q1, k)
        res_glove_200_ids, res_glove_50_scores = glove_200_index.knn_query(q2, k)
        total_res = reduce(np.union1d, (res_dbpedia_ids[0], res_glove_50_ids[0], res_glove_200_ids[0])).tolist()

        combined_scores = []
        for i, res in enumerate(total_res):
            combined_score = w0 * dbpedia_index_flat.get_distance_from(q0, total_res[i]) + \
                     w1 * glove_50_index_flat.get_distance_from(q1, total_res[i]) + \
                     w2 * glove_200_index_flat.get_distance_from(q2, total_res[i])
            combined_scores.append(combined_score)

        T = sorted(combined_scores)[k-1]
        print (f"Threshold is {T}")

        # measure results that were found by this point
        actual_res_ids = [res[1] for res in BF_res]
        correct = 0
        for res in total_res:
            for actual_res in actual_res_ids:
                if res == actual_res:
                    correct += 1
                    break

        print(f"Current results found is {correct} from {len(total_res)} results obtained")

        res_dbpedia_ids, res_dbpedia_scores = dbpedia_index.range_query(q0, T)
        res_glove_50_ids, res_glove_50_scores = glove_50_index.range_query(q1, T)
        res_glove_200_ids, res_glove_200_scores = glove_200_index.range_query(q2, T)
        total_res = reduce(np.union1d, (res_dbpedia_ids[0], res_glove_50_ids[0], res_glove_200_ids[0])).tolist()
        print(f"total res size is {len(total_res)}")



if __name__ == '__main__':
    main()
