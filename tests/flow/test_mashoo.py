import os
os.environ['DOWNLOAD_MULTILANG_DATASET'] = 'false'
os.environ['VERIFY_MULTILANG_DATASET'] = 'false'

# from download_dataset import file_base_name, num_vectors_train, num_vectors_test # if download_dataset() is not commented it will run
num_vectors_train = 10_000_000
num_vectors_test = 10_000
file_base_name = f"multilang_n_{num_vectors_train}_q_{num_vectors_test}"
import pickle
import time
vectors_file_name = f"{file_base_name}_emb.pik"

def split_pickle_file(vectors_file_name, splits):
    with open(vectors_file_name,'rb') as f:
        print(f"loading vectors files from {vectors_file_name}")
        unpickled_array = pickle.load(f)
        print('Array shape: ' + str(unpickled_array.shape))
        print('Data type: '+str(type(unpickled_array)))

        vectors_data = unpickled_array[:num_vectors_train]
        queries_data = unpickled_array[num_vectors_train:num_vectors_train + num_vectors_test]
        assert len(vectors_data) == num_vectors_train
        assert len(queries_data) == num_vectors_test

        batch_size = num_vectors_train // splits
        if num_vectors_train % splits != 0:
            batch_size += 1
        curr_idx = 0

        only_vecs_file_name = f"multilang_n_{num_vectors_train}_emb"

        for split in range(splits):
            with open(f"{only_vecs_file_name}_split_{split}.pik", 'wb') as f:
                pickle.dump(vectors_data[curr_idx:curr_idx + batch_size], f)

        # Save queries data
        only_queries_file_name = f"multilang_q_{num_vectors_test}_emb"
        with open(f"{only_queries_file_name}.pik", 'wb') as f:
            pickle.dump(queries_data, f)


def get_vector_file_count():
    splits = 0
    while os.path.exists(f"{only_vecs_file_name}_split_{splits}.pik"):
        splits += 1
    return splits

def open_pickled_file(filename):
    with open(filename,'rb') as f:
        print(f"loading {filename}")
        unpickled_array = pickle.load(f)
        print('Array shape: ' + str(unpickled_array.shape))
        print('Data type: '+str(type(unpickled_array)))

        return unpickled_array
only_vecs_file_name = f"multilang_n_{num_vectors_train}_emb"

import numpy as np
def check_queries():
    queries_data = open_pickled_file(f"multilang_q_{num_vectors_test}_emb.pik")
    queries_from_all_vectors_data = open_pickled_file(vectors_file_name)[num_vectors_train:num_vectors_train + num_vectors_test]
    assert len(queries_data) == len(queries_from_all_vectors_data)
    assert len(queries_data) == num_vectors_test
    print(f"queries_file_shape: {queries_data.shape}")
    print(f"queries_from_all_vectors_file_shape: {queries_from_all_vectors_data.shape}")
    assert np.array_equal(queries_data, queries_from_all_vectors_data)

def timed_populate_index(index= None, num_vectors = num_vectors_train):
    batches_count = get_vector_file_count()
    total_time = 0
    total_vectors = 0
    sum_vec_sanity = 0
    for i in range(batches_count):
        filename = f"{only_vecs_file_name}_split_{i}.pik"
        with open(filename,'rb') as f:
            print(f"loading {filename}")
            vectors_data = pickle.load(f)
            print('Array shape: ' + str(vectors_data.shape))
            print('Data type: '+str(type(vectors_data)))
            len_vectors_data = len(vectors_data)

            # limit the number of vectors to num_vectors
            if total_vectors + len_vectors_data > num_vectors:
                len_vectors_data = num_vectors - total_vectors
            start_time = time.time()  # Start timing
            for i, vector in enumerate(vectors_data[:len_vectors_data]):
                sum_vec_sanity += 1
            end_time = time.time()  # End timing
            total_time += end_time - start_time
            total_vectors += len_vectors_data
            print(f"Batch {i}: vectors: {len_vectors_data} took: T{end_time - start_time:.4f} seconds")
            print(f"expected {total_vectors} vectors, sanity check: {sum_vec_sanity}")

file_name_prefix = "multilang"
def create_ground_truth_file_name(num_vectors, num_queries):
    return f"{file_name_prefix}_n_{num_vectors}_q_{num_queries}_gt.npy"

def check_gt(num_vectors=num_vectors_train, num_queries=num_vectors_test):
    gt_file_name = create_ground_truth_file_name(num_vectors, num_queries)
    print(f"loading ground truth file from {gt_file_name}")
    my_gt = np.load(gt_file_name, allow_pickle=True).item()
    queries_data = open_pickled_file(f"multilang_q_{num_queries}_emb.pik")
    print(f"my queries data len: {len(queries_data)}")

    # print("my_gt[0]", my_gt[1])
    print("type(my_gt.items()[1])", type(my_gt))
    omer_gt_prefix = "/home/ubuntu/VectorSimilarity/ground_truth"

    for i in range(num_queries):
        gt_labels, gt_distances = my_gt[i]
        omer_gt_labels_file_name = f"{omer_gt_prefix}/ids{i}.npy"

        omer_gt = np.load(omer_gt_labels_file_name)
        if not np.array_equal(gt_labels[0], omer_gt[0]):
            print(f"gt_labels for query {i} are not equal")
            for j, label in enumerate(gt_labels[0]):
                if label != omer_gt[0][j]:
                    print(f"label {j} is diffrent. my_gt: {label}, omer_gt: {omer_gt[0][j]}")
                    print(f"distance is: {gt_distances[0][j]}")
            print()

        # omer_gt_vector_file_name = f"{omer_gt_prefix}/vector{i}.npy"
        # try:
        #     omer_query = np.load(omer_gt_vector_file_name)
        #     print(f"omer query: {omer_query}")
        # except EOFError as e:
        #     print("EOFError at query: ", i)
        # if not np.array_equal(queries_data[i], omer_query[0]):
        #     print(f"queries_data for query {i} are not equal")
        #     for j, val in enumerate(queries_data[i]):
        #         if val != omer_query[0][j]:
        #             print(f"val {j} is diffrent. my_gt: {val}, omer_gt: {omer_query[0][j]}")
        #     print()
        # try:
        #     assert np.array_equal(queries_data[i], np.load(omer_gt_vector_file_name)), f"query {i} is not equal"
        # except EOFError as e:
        #     print("EOFError at query: ", i)

check_gt()
# split_pickle_file(vectors_file_name, 5)
# timed_populate_index()
# check_queries()
