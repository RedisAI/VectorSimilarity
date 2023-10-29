# Vector Similarity benchmarks user guide

## Table of contents
* [Overview](#overview)
* [Run benchmarks](#run-benchmarks)
* [Available sets](#available-sets)  
    - [BM_VecSimBasics](#bm_vecsimbasics)  
    - [BM_BatchIterator](#bm_batchiterator)  
    - [BM_VecSimUpdatedIndex](#bm_vecsimupdatedindex)  
* [ann-benchmarks](#ann-benchmark)

# Overview
We use the [Google benchmark](https://github.com/google/benchmark) tool to run micro-benchmarks for the vector indexes functionality.  
Google benchmark is a popular tool for benchmark code snippets, similar to unit tests. It allows a quick way to estimate the runtime of each use case based on several (identical) runs, and prints the results as output. For some tests, the output includes an additional "Recall" metric, which indicates the accuracy in the case of approximate search.  
**The recall** is calculated as the size of the intersection set between the number of the ground truth results (calculated by the flat algorithm) and the returned results from the approximate algorithm (HNSW in this case), divided by the number of ground truth results:  
$$ recall = \frac{\text{approximate algorithm results } \cap
\text{ ground truth results } } {\text{ground truth results}}
$$
# Run benchmarks
## Required files
The serialized indices files that are used for micro-benchmarking and running ann-benchmark can be found in
`tests/benchmark/data/hnsw_indices.txt`.  
To download all the required files, run from the repository root directory:
```sh
wget --no-check-certificate -q -i tests/benchmark/data/hnsw_indices/hnsw_indices_all.txt -P tests/benchmark/data
```
To run all test sets, call the following commands from the project root dir:
```sh
make benchmark
```
### Running a Subset of Benchmarks
To run only a subset of benchmarks that match a specified `<regex>`, set `BENCHMARK_FILTER=<regex>` environment variable. For example:  
```sh
make benchmark BENCHMARK_FILTER=fp32*
```

# Available sets
There are currently 3 sets of benchmarks available: `BM_VecSimBasics`, `BM_BatchIterator`, and `BM_VecSimUpdatedIndex`. Each is templated according to the index data type. We run 10 iterations of each test case, unless otherwise specified.
## BM_VecSimBasics
For each combination of data type (fp32/fp64), index type (single/multi), and indexing algorithm (flat, HNSW and tiered-HNSW) the following test cases are included:
1. Measure index total `memory` (runtime and iterations number are irrelevant for this use case, just the memory metric) 
2. `AddLabel` - runs for `DEFAULT_BLOCK_SIZE (= 1024)` iterations, in each we add one new label to the index from the `queries` list. Note that for a single value index each label contains one vector, meaning that the number of new labels equals the number of new vectors.  
**results:** average time per label, average memory addition per vector, vectors per label.  
*At the end of the benchmark, we delete the added labels*
3. `DeleteLabel` - runs for `DEFAULT_BLOCK_SIZE (= 1024)` iterations, in each we delete one label from the index. Note that for a single value index each label contains one vector, meaning that the number of deleted labels equals the number of deleted vectors.  
**results:** average time per label, average memory addition per vector (a negative value means that the memory has decreased).  
*At the end of the benchmark, we restore the deleted vectors under the same labels*

For tiered-HNSW index, we also run these additional tests:

4. `AddLabel_Async` - which is the same as `AddLabel` test, but here we also take into account the time that it takes to the background threads to ingest vectors into HNSW asynchronously (while in `AddLabel` test for the tiered index we only measure the time until vectors are stored in the flat buffer).
5. `DeleteLabel_Async` - which is the same as `DeleteLabel` test, but here we also take into account the time that it takes to the background threads to repair the HNSW graph due to the deletion (while in `DeleteLabel` test for the tiered index we only measure the time until vectors are marked as deleted). Note that the garbage collector of the tiered index is triggered when at least `swapJobsThreshold` vectors are ready to be evicted (this happens when all of their affected neighbors in the graph are repaired). We run this test for `swapJobsThreshold` in `{1, 100, 1024(DEFAULT)}`, and we collect two additional metrics: `num_zombies`, which is the number of vectors that are left to be evicted *after* the test has finished, and `cleanup_time`, which is the number of milliseconds that it took to clean these zombies.

In both tests, we should only consider the `real_time` metric (rather than the `cpu_time`), since `cpu_time` only accounts for the time that the main thread is running. 
#### **TopK benchmarks**
Search for the `k` nearest neighbors of the query.   
6. Run `Top_K` query over the flat index (using brute-force search), for each `k=10`, `k=100` and `k=500`  
**results:** average time per iteration  
7. Run `Top_K` query over the HNSW index, for each pair of `{ef_runtime, k}` from the following:  
    `{ef_runtime=10, k=10}`   
    `{ef_runtime=200, k=10}`   
    `{ef_runtime=100, k=100}`   
    `{ef_runtime=200, k=100}`   
    `{ef_runtime=500, k=500}`   
**results:** average time per iteration, recall
8. Run `Top_K` query over the tiered-HNSW index (in parallel by several threads), for each pair of `{ef_runtime, k}` as in the previous test (assuming all vector are indexed into the HNSW graph). We run for `50` iterations to get a better sense of the parallelism that can be achieved. Here as well, we should only consider the `real_time` measurement rather than the `cpu_time`.
#### **Range query benchmarks**
In range query, we search for all the vectors in the index whose distance to the query vector is lower than the range.  
9. Run `Range` query over the flat index (using brute-force search), for each `radius=0.2`, `radius=0.35` and `radius=0.5`  
**results:** average time per iteration, average results number per iteration  
10. Run `Range` query over the HNSW index, for each pair of `{radius, epsilon}` from the following:  
    `{radius=0.2, epsilon=0.001}`   
    `{radius=0.2, epsilon=0.01}`    
    `{radius=0.2, epsilon=0.1}`     
    `{radius=0.35, epsilon=0.001}`   
    `{radius=0.35, epsilon=0.01}`      
    `{radius=0.35, epsilon=0.1}`      
    `{radius=0.5, epsilon=0.001}`   
    `{radius=0.5, epsilon=0.01}`   
    `{radius=0.5, epsilon=0.1}`   
**results:** average time per iteration, average results number per iteration, recall  

## BM_BatchIterator
The purpose of these tests is to benchmark batch iterator functionality. The batch iterator is a handle that enables running `Top_K` query in batches, by asking for the next best `n` results repeatedly, until there are no more results to return. We use for this test cases the same indices that were built for the "basic benchmark" for this test case as well.  
The test cases are:
1. Fixed batch size - Run `Top_K` query for each pair of `{batch size, number of batches}` from the following:  
`{batch size=10, number of batches=1}`  
`{batch size=10, number of batches=3}`  
`{batch size=10, number of batches=5}`  
`{batch size=100, number of batches=1}`  
`{batch size=100, number of batches=3}`  
`{batch size=100, number of batches=5}`  
`{batch size=1000, number of batches=1}`  
`{batch size=1000, number of batches=3}`  
`{batch size=1000, number of batches=5}`  
**Flat index results:** Time per iteration, memory delta per iteration
**HNSW index results:** Time per iteration,  Recall, memory delta per iteration
2. Variable batch size - Run `Top_K` query where in each iteration the batch size is increased by a factor of 2, for each pair of `{batch initial size, number of batches}` from the following:  
`{batch initial size=10, number of batches=2}`  
`{batch initial size=10, number of batches=4}`  
`{batch initial size=100, number of batches=2}`  
`{batch initial size=100, number of batches=4}`  
`{batch initial size=1000, number of batches=2}`  
`{batch initial size=1000, number of batches=4}`  
**Flat index results:** Time per iteration
**HNSW index results:** Time per iteration, Recall, memory delta per iteration
3. Batches to Adhoc BF - In each iteration we run `Top_K` with an increasing `batch size` (initial size=10, increase factor=2) for `number of batches` and then switch to ad-hoc BF. We define `step` as the ratio between the index size to the number of vectors to go over in ad-hoc BF. The tests run for each pair of `{step, number of batches}` from the following:  
`{step=5, number of batches=0}`  
`{step=5, number of batches=2}`  
`{step=5, number of batches=5}`  
`{step=10, number of batches=0}`  
`{step=10, number of batches=2}`  
`{step=10, number of batches=5}`  
`{step=20, number of batches=0}`  
`{step=20, number of batches=2}`  
`{step=20, number of batches=5}`  
**Flat index results:** Time per iteration
**HNSW index results:** Time per iteration, memory delta per iteration

## BM_VecSimUpdatedIndex
For this use case, we create two indices for each algorithm (flat and HNSW). The first index contains 500K vectors added to an empty index. The other index also contains 500K vectors, but this time they were added by overriding the 500K vectors of the aforementioned indices. Currently, we only test this for FP32 single-value index.  
The test cases are:
1. Index `total memory` **before** updating
2. Index `total memory` **after** updating
3. Run `Top_K` query over the flat index **before** updating (using brute-force search), for each `k=10`, `k=100` and `k=500`  
**results:** average time per iteration
4. Run `Top_K` query over the flat index **after** updating (using brute-force search), for each `k=10`, `k=100` and `k=500`  
**results:** average time per iteration
5. Run **100** iterations of `Top_K` query over the HNSW index **before** updating, for each pair of `{ef_runtime, k}` from the following:  
    `{ef_runtime=10, k=10}`   
    `{ef_runtime=200, k=10}`   
    `{ef_runtime=100, k=100}`   
    `{ef_runtime=200, k=100}`   
    `{ef_runtime=500, k=500}`   
**results:** average time per iteration, recall
6. Run **100** iterations of `Top_K` query over the HNSW index **after** updating, for each pair of `{ef_runtime, k}` from the following: 
    `{ef_runtime=10, k=10}`   
    `{ef_runtime=200, k=10}`   
    `{ef_runtime=100, k=100}`   
    `{ef_runtime=200, k=100}`   
    `{ef_runtime=500, k=500}`   
**results:** average time per iteration, recall  

# ANN-Benchmark

[ANN-Benchmarks](http://ann-benchmarks.com/) is a benchmarking environment for approximate nearest neighbor algorithms search (for additional information, refer to the project's [GitHub repository](https://github.com/erikbern/ann-benchmarks)).  Each algorithm is benchmarked on pre-generated commonly use datasets (in HDF5 formats).
The `bm_dataset.py` script uses some of ANN-Benchmark datasets to measure this library performance in the same manner. The following datasets are downloaded and benchmarked (all use FP32 single value per label data):

1. glove-25-angular
2. glove-50-angular
3. glove-100-angular
4. glove-200-angular
5. mnist-784-euclidean
6. sift-128-euclidean

For each dataset, the script will build an HNSW index with pre-defined build parameters and persist it to a local file in `./data` directory that will be generated (index file name for example: `glove-25-angular-M=16-ef=100.hnsw`). Note that if the file already exists in this path, the entire index will be loaded instead of rebuilding it. 
To download the serialized indices run from the project's root directory:
```sh
wget -q -i tests/benchmark/data/hnsw_indices_all/hnsw_indices_ann.txt -P tests/benchmark/data
```
Then, for 3 different pre-defined `ef_runtime` values, 1000 `Top_K` queries will be executed for `k=10` (these parameters can be modified easily in the script). For every configuration, the script outputs the following statistics:

- Average recall
- Query-per-second when running in brute-force mode
- Query-per-second when running with HNSW index
To reproduce this benchmark, first install the project's python bindings, and then invoke the script. From the project's root directory, you should run:
```py
python3 tests/benchmark/bm_datasets.py
```