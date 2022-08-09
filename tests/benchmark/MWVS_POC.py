import os
from urllib.request import urlretrieve
import numpy as np
import time
from VecSim import *
import h5py


def create_indexes(params, index_file_name):
    hnsw_index = HNSWIndex(params)
    hnsw_index.load_index(index_file_name)

    bfparams = BFParams()
    bfparams.initialCapacity = params.initialCapacity
    bfparams.dim = params.dim
    bfparams.type = VecSimType_FLOAT32
    bfparams.metric = VecSimMetric_Cosine
    flat_index = BFIndex(bfparams)
    return hnsw_index, flat_index


def load_test_set(dateset):
    file_name = "tests/benchmark/data/"+dateset
    if dateset != "dbpedia-768":
        file_name += "-angular"
    file_name += ".hdf5"
    return np.array(h5py.File(file_name, 'r')['test'])


def main():
    # Create 3 HNSW indexes:
    # DBPedia #
    hnswparams = HNSWParams()
    hnswparams.M = 64
    hnswparams.efConstruction = 512
    hnswparams.initialCapacity = 1000000
    hnswparams.efRuntime = 10
    hnswparams.dim = 768
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine
    dbpedia_index, dbpedia_index_flat =\
        create_indexes(hnswparams, "tests/benchmark/data/DBpedia-n1M-cosine-d768-M64-EFC512.hnsw_v1")

    # Glove-100 #
    hnswparams = HNSWParams()
    hnswparams.M = 36
    hnswparams.efConstruction = 250
    hnswparams.initialCapacity = 1183514
    hnswparams.efRuntime = 300
    hnswparams.dim = 100
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine
    glove_100_index, glove_100_index_flat =\
        create_indexes(hnswparams, "tests/benchmark/data/glove-100-angular-M=36-ef=250.hnsw")

    # Glove-200 #
    hnswparams = HNSWParams()
    hnswparams.M = 48
    hnswparams.efConstruction = 350
    hnswparams.initialCapacity = 1183514
    hnswparams.efRuntime = 350
    hnswparams.dim = 200
    hnswparams.type = VecSimType_FLOAT32
    hnswparams.metric = VecSimMetric_Cosine
    glove_200_index, glove_200_index_flat =\
        create_indexes(hnswparams, "tests/benchmark/data/glove-200-angular-M=48-ef=350.hnsw")

    print("3 Indexes loaded successfully")

    # Run multi top k search
    test_sets = {}
    data_sets = ["dbpedia-768", "glove-100", "glove-200"]
    for data_set in data_sets:
        test_sets[data_set] = load_test_set(data_set)
        print(f"loaded test set for {data_set} of size {len(test_sets[data_set])}")

    for i in range(1000):
        q_0 = test_sets[data_sets[0]][i]
        q_1 = test_sets[data_sets[1]][i]
        q_2 = test_sets[data_sets[2]][i]

        # todo: cont from here algo implementation


if __name__ == '__main__':
    main()
