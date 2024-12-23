from datasets import load_dataset
import numpy as np
import os
import h5py
from math import ceil
DOCS_LIM = 10**6 # 1 million
TEST_LIM = 10**4 # 
INT8_KEY = 'emb_int8'
DATASET = 'wikipedia-1024_eng_v3'
hdf5_output_file_name = "%s.hdf5" %DATASET

lang = "en" #Use the English Wikipedia subset
docs = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3-int8-binary", lang, split="train")

train = docs.select(range(DOCS_LIM))
test = docs.select(range(DOCS_LIM,DOCS_LIM+TEST_LIM))
doc = docs[0]
# dataset = np.array(docs,dtype='int8')
int8_emb = np.array(doc[INT8_KEY],dtype='int8')
# dataset_array = np.array(docs,dtype=np.int8)

# dataset in 7Gb is size, so need to buffer
# divide into chunks of 2**14
CHUNK_SIZE = 2**14
n_chunks = ceil(DOCS_LIM/CHUNK_SIZE)
with h5py.File(hdf5_output_file_name,"w") as f:
    dset = f.create_dataset("train",(CHUNK_SIZE,int8_emb.shape[0]),maxshape = (None,int8_emb.shape[0]),dtype='i8')
    for i in range(1,n_chunks+1):
        print(f'train: writing chunk {i}/{n_chunks}...',end='')
        if i>0 and i<n_chunks:
            dset.resize(min(dset.shape[0]+CHUNK_SIZE,len(train)),axis=0)
        start_index = max(0,CHUNK_SIZE*(i-1))
        end_index =  min(len(train),CHUNK_SIZE*i)
        chunk = train.select(range(start_index,end_index))
        chunk_arr = np.array([doc[INT8_KEY] for doc in chunk],dtype=np.int8)
        dset[start_index - end_index:] = chunk_arr
        # write to file
        del chunk_arr
        print('done!')
    n_chunks = ceil(TEST_LIM/CHUNK_SIZE)
    dset = f.create_dataset("test",(min(CHUNK_SIZE,len(test)),int8_emb.shape[0]),maxshape = (None,int8_emb.shape[0]),dtype='i8')
    for i in range(1,n_chunks+1):
        print(f'test: writing chunk {i}/{n_chunks}...',end='')
        if i>0 and i<n_chunks:
            dset.resize(min(dset.shape[0]+CHUNK_SIZE,len(test)),axis=0)
        start_index = max(0,CHUNK_SIZE*(i-1))
        end_index =  min(len(test),CHUNK_SIZE*i)
        chunk = test.select(range(start_index,end_index))
        chunk_arr = np.array([doc[INT8_KEY] for doc in chunk],dtype=np.int8)
        dset[start_index - end_index:] = chunk_arr
        # write to file
        del chunk_arr
        print('done!')
print('Writing file completed')