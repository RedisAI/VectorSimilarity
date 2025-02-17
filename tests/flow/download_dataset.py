from datasets import load_dataset
import numpy as np
import pickle
import time
import os

DOWNLOAD_DATASET = False
VERIFY_DATASET = False
env_var = os.environ.get('DOWNLOAD_MULTILANG_DATASET')
if env_var == 'false':
    DOWNLOAD_DATASET = False
if env_var == 'true':
    DOWNLOAD_DATASET = True

env_var = os.environ.get('VERIFY_MULTILANG_DATASET')
if env_var == 'false':
    VERIFY_DATASET = False
if env_var == 'true':
    VERIFY_DATASET = True

lang = "en" #Use the Simple English Wikipedia subset
num_vectors_train = 2000
# num_vectors_train = 10_000_000
# num_vectors_test = 10_000
num_vectors_test = 10
num_vectors = num_vectors_train + num_vectors_test

dim = 1024
docs = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", lang, split="train", streaming=True)
vecs = np.zeros((num_vectors, dim), dtype=np.float32)  # Use float32 for memory efficiency

fields = ['emb', '_id', 'text', 'title']
# files format: multilang_n_{num_vectors_train}_q_{num_vectors_test}_{field}.pik

# dict of <field>: <data>
data = {}

file_base_name = f"multilang_n_{num_vectors_train}_q_{num_vectors_test}"
def download_dataset():
    should_download = 'y'
    for field in fields:
        file = f"{file_base_name}_{field}.pik"
        if os.path.exists(file):
            should_download = input(f"{field} file exists. Should override? (y/n)")
        else:
            should_download = input(f"{field} file does not exist. Should we create it? (y/n)")
        if should_download.lower() == 'y':
            print(f"Downloading {field} data to {file}")
            if field == 'emb':
                data[field] = vecs
            else:
                data[field] = []

    if data == {}:
        print("Nothing to download")
        return

    counter = 0
    start_time = time.time()  # Start timing
    for doc in docs:
        if counter == num_vectors:
            break
        for key in data.keys():
            if key == 'emb':
                vecs[counter] = doc[key] # add to pre-allocated numpy array
            else:
                data[key].append(doc[key]) # add to meta data list

        counter += 1
    end_time = time.time()  # End timing
    print('load time: ',f"T{end_time - start_time:.4f} seconds")
    start_time = time.time()  # Start timing

    for key in data.keys():
        with open(f"{file_base_name}_{key}.pik", 'wb') as f:
            pickle.dump(data[key], f)

def load_dataset_from_disk():
    file = f"{file_base_name}_emb.pik"
    with open(file,'rb') as f:
        unpickled_array = pickle.load(f)
        # dim = unpickled_array.shape[1]
        print('Array shape: ' + str(unpickled_array.shape))
        print('Data type: '+str(type(unpickled_array)))

def verify_downloaded_dataset():
    for field in fields:
        file = f"{file_base_name}_{field}.pik"
        if os.path.exists(file):
            with open(file,'rb') as f:
                unpickled_array = pickle.load(f)
                if field == 'emb':
                    assert unpickled_array.shape == (num_vectors, dim)
                    for i in range(num_vectors):
                        assert np.any(unpickled_array[i]), f"Array at index {i} is all zeros"
                elif field == '_id':
                    assert len(unpickled_array) == num_vectors
                elif field == 'text':
                    assert len(unpickled_array) == num_vectors
                elif field == 'title':
                    assert len(unpickled_array) == num_vectors
                print(f"{field} - ok")

if DOWNLOAD_DATASET == True:
    download_dataset()
if VERIFY_DATASET == True:
    verify_downloaded_dataset()
verify_downloaded_dataset()
# load_dataset_from_disk()
#
