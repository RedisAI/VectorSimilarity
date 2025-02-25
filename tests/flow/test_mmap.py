
# use to disable automatic dataset download in download_dataset.py
import os
os.environ['DOWNLOAD_MULTILANG_DATASET'] = 'false'
os.environ['VERIFY_MULTILANG_DATASET'] = 'false'

# from download_dataset import file_base_name, num_vectors_train, num_vectors_test # if download_dataset() is not commented it will run
num_vectors_train = 10_000_000
num_vectors_test = 10_000
file_base_name = f"multilang_n_{num_vectors_train}_q_{num_vectors_test}"

import pickle
from VecSim import *
from common import create_hnsw_index, create_flat_index
import hnswlib
from numpy.testing import assert_allclose
import numpy as np
import time
import json
import psutil

# To check file size at runtime look for the smallest fd that is marked as deleted
# this is level 0 vectors file.
# run stat /proc/283956/fd/<fd> --dereference
# Than divide it by level 0 elemnt data size to get the number of current indexedvectors
# for example, M = 60, (M0 = 120):
# level0 data size: 2 * M * sizeof(id_type) + sizeof(metadata_struct) + sizeof(num_liks_data_type)
# 2 * 60 * 4 + 8 + 2 = 484
# divide it by 490 (bytes per vector)

# this one works as well:
# sudo lsof -p `pgrep pytest` | awk '$4 ~ /^[0-9]+u$/'

# Globals
RUN_SANITY = False
RUN_BM = True
RUN_GT = False

MAX_K = 100
MMAP_ADVISE = "MADV_DONTNEED"
ENABLE_LOGS = False
LOAD_ALL_VECTORS = True
file_name_prefix = "multilang"
all_vectors_file_name = f"{file_base_name}_emb.pik"
only_vecs_file_name = f"multilang_n_{num_vectors_train}_emb"
queries_file_name = f"multilang_q_{num_vectors_test}_emb.pik"

classic_print = print
def print(*args, **kwargs):
    kwargs['flush'] = True
    classic_print(f'[{time.time()}]', *args, **kwargs)

def get_rss_memory_usage_bytes():
    process = psutil.Process()
    return process.memory_info().rss  # RSS in bytes

def create_ground_truth_file_name(num_vectors, num_queries):
    return f"{file_name_prefix}_n_{num_vectors}_q_{num_queries}_gt.npy"

def open_pickled_file(filename, mode='rb'):
    with open(filename, 'rb') as f:
        print(f"loading {filename}")
        data = pickle.load(f)
        print('Array shape: ' + str(data.shape))
        print('Data type: '+str(type(data)))

    return data

queries_data = open_pickled_file(queries_file_name)
dim = queries_data.shape[1]
if LOAD_ALL_VECTORS:
    vectors_data = open_pickled_file(all_vectors_file_name)

def get_vector_file_count():
    splits = 0
    while os.path.exists(f"{only_vecs_file_name}_split_{splits}.pik"):
        splits += 1
    return splits

def timed_populate_index(index, num_vectors, check_memory_interval=0):
    check_memory_interval = num_vectors // 10 if check_memory_interval == 0 else check_memory_interval

    batches_count = get_vector_file_count()
    build_time = 0
    total_vectors = 0
    done = False
    for i in range(batches_count):
        filename = f"{only_vecs_file_name}_split_{i}.pik"
        with open(filename,'rb') as f:
            if done == True:
                break
            print(f"loading {filename}")
            vectors_data = pickle.load(f)
            print('Array shape: ' + str(vectors_data.shape))
            print('Data type: '+str(type(vectors_data)))
            len_vectors_data = len(vectors_data)

            # limit the number of vectors to num_vectors
            if total_vectors + len_vectors_data > num_vectors:
                len_vectors_data = num_vectors - total_vectors
                done = True

            start_time = time.time()  # Start timing
            for vector in vectors_data[:len_vectors_data]:
                index.add_vector(vector, total_vectors)
                total_vectors += 1

                if total_vectors % check_memory_interval == 0:
                    end_time = time.time()
                    build_time += end_time - start_time
                    print(f"Building {total_vectors} vectors time: ", f"{build_time:.4f} seconds")
                    curr_mem = index.index_memory()
                    print(f"Current memory usage: {curr_mem} bytes, {(curr_mem / 1024 / 1024):.4f} MB, {(index.index_memory() / 1024 / 1024 / 1024):.4f} GB")
            end_time = time.time()  # End timing
            build_time += end_time - start_time
            print(f"Batch {i}: vectors: {len_vectors_data} took: {(end_time - start_time):.4f} seconds")

            print(f"printing a vec: {vectors_data[0]}")

    return build_time


def build_grount_truth(num_vectors=num_vectors_train, num_queries=num_vectors_test, dim=1024):
    index = create_flat_index(dim, VecSimMetric_L2, VecSimType_FLOAT32)
    print("\nBuilding Flat index")
    # build_time = timed_populate_index(index, num_vectors)
    start_time = time.time()  # Start timing
    for i, vector in enumerate(vectors_data[:num_vectors]):
        index.add_vector(vector, i)

    end_time = time.time()
    build_time = end_time - start_time
    print('Building time: ',f"{build_time:.4f} seconds")

    print(f"Get {MAX_K} ground truth vectors for each query vector")

    knn_results = {}
    for i, query_vector in enumerate(queries_data[:num_queries]):
        labels, distances = index.knn_query(query_vector, k=MAX_K)
        knn_results[i] = (labels, distances)

    ground_truth_file_name = create_ground_truth_file_name(num_vectors, num_queries)
    np.save(ground_truth_file_name, knn_results, allow_pickle=True)

if RUN_GT:
    build_grount_truth(num_vectors=1_000_000)

def load_gt(num_vectors, num_queries):
    ground_truth_file_name = create_ground_truth_file_name(num_vectors, num_queries)
    print(f"loading {ground_truth_file_name}")
    return np.load(ground_truth_file_name, allow_pickle=True).item()

def write_result_to_file(result, filename="results.json", override=False):
    print(f"writing results to file {filename}")
    mode = 'a' if not override else 'w'
    try:
        with open(filename, mode) as f:
            json.dump(result, f, indent=4)
            f.write('\n')  # Write each result on a new line
    except Exception as e:
        print(f"Failed to write result to file: {e}")

def bm_query(index, k, efR, gt_results, num_queries=num_vectors_test):
    print(f"\nRunning {num_queries} queries benchmark with params: efR: {efR}, k: {k}")
    index.set_ef(efR)
    total_query_time = 0
    recall = 0
    total_recall = 0
    for i, query_data in enumerate(queries_data[:num_queries]):
        start_time = time.time()
        hnsw_labels, hnsw_distances = index.knn_query(query_data, k=k)
        end_time = time.time()
        total_query_time += end_time - start_time

        # compute recall
        gt_labels, gt_distances = gt_results[i]
        recall = set(hnsw_labels.flatten()).intersection(set(gt_labels.flatten()))
        # print(f"hnsw_labels.flatten(): {hnsw_labels.flatten()}")
        # print(f"gt_labels.flatten(): {gt_labels.flatten()}")
        recall = float(len(recall)) / float(k)
        total_recall += recall
        if i % 1000 == 0:
            print(f"Query {i}: recall={recall}, time={total_query_time:.4f} seconds")

    avg_query_time = total_query_time / num_queries
    avg_recall = total_recall / num_queries

    result = {
        "num_queries": num_queries,
        "k": k,
        "efR": efR,
        "total_query_time": total_query_time,
        "avg_query_time": avg_query_time,
        "avg_recall": avg_recall
    }

    return result


def bm_test_case(M, efC, Ks_efR, num_vectors=num_vectors_train, num_queries=num_vectors_test):
    index = create_hnsw_index(
        dim,
        num_vectors,
        VecSimMetric_L2,
        VecSimType_FLOAT32,
        m=M,
        ef_construction=efC,
    )

    if ENABLE_LOGS == False: index.disable_logs()

    index_block_size = 'None'
    print(f"\nBuilding index of size {num_vectors:,} with params: ", f"M: {M}, efC: {efC}, index_block_size: {index_block_size}, madvise: {MMAP_ADVISE} \n", flush=True)

    check_memory_interval = num_vectors // 50

    if LOAD_ALL_VECTORS == False: open_pickled_file(all_vectors_file_name)
    build_time = 0
    start_time = time.time()  # Start timing
    for i, vector in enumerate(vectors_data[:num_vectors]):
        index.add_vector(vector, i)

        if i % check_memory_interval == 0:
            end_time = time.time()
            build_time += end_time - start_time
            print(f"Building {i} vectors time: ", f"{build_time:.4f} seconds")
            curr_mem = index.index_memory()
            print(f"Current index memory usage: {curr_mem} bytes, {(curr_mem / 1024 / 1024):.4f} MB, {(index.index_memory() / 1024 / 1024 / 1024):.4f} GB")

            proc_rss = get_rss_memory_usage_bytes()
            print(f"Current process RSS memory usage: {proc_rss} bytes, {(proc_rss / 1024 / 1024):.4f} MB, {(proc_rss / 1024 / 1024 / 1024):.4f} GB")
            start_time = time.time()

    print('\nBuilding time: ', f"{build_time:.4f} seconds, {(build_time / 60):.4f} m, {(build_time / 60 / 60):.4f} h\n")
    index_max_level = index.index_max_level()
    print(f"index_max_level: {index_max_level}")

    # Sanity checks
    print(f"disk hnsw index contains {(index.index_size()):,} vectors")

    random_query_index = np.random.randint(0, num_vectors)

    # query with a vector from the dataset
    index_query_data = index.get_vector(random_query_index)[0]
    query_data = vectors_data[random_query_index]
    assert np.array_equal(query_data, index_query_data)
    print("query_data is equal to index_query_data")

    # expect to get the same vector back with distance of 0
    labels, distances = index.knn_query(query_data, k=1)
    print(f"testing vector: {random_query_index}")
    print("labels: ", labels)
    print("distances: ", distances)
    sanity_checks = {
        "index_size": "Pass" if (index.index_size() == num_vectors) else "Fail",
        "query_label_result": "Pass" if (labels[0][0] == random_query_index) else "Fail",
        "distance_check": "Pass" if (distances[0][0] == float(0)) else "Fail",
    }
    failure_info = {}


    if sanity_checks["index_size"] == "Fail":
        failure_info["index_size"] = {
            "expected": index.index_size(),
            "actual": num_vectors
        }

    if sanity_checks["query_label_result"] == "Fail":
        failure_info["query_label_result"] = {
            "expected": int(labels[0][0]),
            "actual": random_query_index
        }

    if sanity_checks["distance_check"] == "Fail":
        failure_info["distance_check"] = {
            "expected": distances[0][0],
            "actual": float(0)
        }
    index_settings = {
        "mmap_advise": MMAP_ADVISE,
        "index_block_size": index_block_size,
    }

    build_result = {
        "M": M,
        "efC": efC,
        "num_vectors": num_vectors,
        "build_time": build_time,
        "max_level": index_max_level,
        "sanity_checks": sanity_checks,
        "failure_info": failure_info,
    }

    gt_results = load_gt(num_vectors, num_queries)
    knn_bm_results = []
    for k, efR in Ks_efR:
        queries_reslts = bm_query(index, k=k, efR=efR, gt_results=gt_results)
        knn_bm_results.append(queries_reslts)

    result = {
        "index_settings": index_settings,
        "build_bm": build_result,
        "knn_bm_results": knn_bm_results,
    }

    results_file_name = f"results_M_{M}_efC_{efC}_vec_{num_vectors}_q_{num_queries}_madvise_{MMAP_ADVISE}_bs_{index_block_size}.json"
    write_result_to_file(result, filename=results_file_name, override=True)

def bm():
    input(f"PID: {os.getpid()} Press Enter to continue...")
    # Build params
    Ms_efC = [(60, 75), (120, 150), (150, 150), (200, 120), (200, 150)]

    # query params
    Ks = [1, 10, 100]
    factors = [200, 100, 20, 10, 2]
    factors = [2, 10, 20, 100, 200]
    Ks_efR = []
    max_efR = 200
    for k in Ks:
        for factor in factors:
            if k * factor <= max_efR:
                Ks_efR.append((k, k * factor))

    for M, efC in Ms_efC[:1]:
        bm_test_case(M=M, efC=efC, Ks_efR=Ks_efR)

def sanity_vecsim_mmap():
    efRuntime = 10
    M = 16
    efConstruction = 100

    num_vectors = min(1000, num_vectors_train - 1)

    index = create_hnsw_index(
        dim,
        num_vectors_train,
        VecSimMetric_L2,
        VecSimType_FLOAT32,
        m=M,
        ef_construction=efConstruction,
        ef_runtime=efRuntime,
    )
    print("\ndisk hnsw index created")

    print("Create hnswlib index for sanity testing")
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_vectors_train, ef_construction=efConstruction, M=M)
    p.set_ef(efRuntime)

    vectors_data = open_pickled_file(all_vectors_file_name)
    for i, vector in enumerate(vectors_data[:num_vectors]):
        index.add_vector(vector, i)
        p.add_items(vector, i)

    print(f"disk hnsw index containts {index.index_size()} vectors")
    print(f"hnswlib index containts {p.get_current_count()} vectors")

    print("Testing knn")
    query_data = queries_data[0]
    hnswlib_labels, hnswlib_distances = p.knn_query(query_data, k=10)
    redis_labels, redis_distances = index.knn_query(query_data, 10)
    assert_allclose(hnswlib_labels, redis_labels, rtol=1e-5, atol=0)
    # print(f"redis labels = {redis_labels}, hnswlib labels = {hnswlib_labels}")
    assert_allclose(hnswlib_distances, redis_distances, rtol=1e-5, atol=0)
    # print(f"redis distances = {redis_distances}, hnswlib distances = {hnswlib_distances}")
    print("Testing knn ok")

def gt_vecsim_mmap():
    num_vectors = min(1000, num_vectors_train - 1)
    efRuntime = 10
    M = 16
    efConstruction = 100
    metric = VecSimMetric_L2
    data_type = VecSimType_FLOAT32
    index = create_hnsw_index(
        dim=dim,
        num_elements=num_vectors,
        metric=metric,
        data_type=data_type,
        m=M,
        ef_construction=efConstruction,
        ef_runtime=efRuntime,
    )
    print("\ndisk hnsw index created")

    bf_index = create_flat_index(dim = dim, metric=metric, data_type=data_type)
    print("\ndisk bf_index index created")

    vectors_data = open_pickled_file(all_vectors_file_name)
    for i, vector in enumerate(vectors_data[:num_vectors]):
        index.add_vector(vector, i)
        bf_index.add_vector(vector, i)

    print(f"disk hnsw index contains {index.index_size()} vectors")
    print(f"disk bf index contains {bf_index.index_size()} vectors")

    print("Testing knn")
    query_data = queries_data[0]
    flat_labels, flat_distances = bf_index.knn_query(query_data, k=10)
    hnsw_labels, hnsw_distances = index.knn_query(query_data, 10)

    fail = False
    res = np.allclose(flat_labels, hnsw_labels, rtol=1e-5, atol=0)
    if not res:
        print("Testing knn labels not ok")
        fail = True
    print(f"hnsw labels = {hnsw_labels}, flat labels = {flat_labels}")
    res = np.allclose(flat_distances, hnsw_distances, rtol=1e-5, atol=0)
    if not res:
        print("Testing knn dists not ok")
        fail = True
    print(f"hnsw distances = {hnsw_distances}, flat distances = {flat_distances}")
    assert not fail

def test_sanity():
    if (RUN_SANITY):
        sanity_vecsim_mmap()
        gt_vecsim_mmap()
        gt_vecsim_mmap()

def test_bm():
    if (RUN_BM):
        bm()
