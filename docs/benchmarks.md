# Benchmark

To get an early sense of what VectorSimilarity library by RedisAI can do, you can test it with the following benchmark tools:

## Google benchmark

Google benchmark is a popular tool for benchmark code snippets, similar to unit tests. It allows a quick way of estimating the runtime of each test based on several (identical) runs, and print the results as output. For some tests, the output includes additional "Recall" metric, which indicates the accuracy in case of approximate search.

There are 2 tests suits available: `BM_VecSimBasics`, and `BM_BatchIterator`.
To run both test suits, call the following commands from the project root dir:
```asm
make
make benchmark
```

#### Basic benchmark

In this test suit, we create two indices that contains (the same) 1M random vectors of 100 floats - first is a "Flat" index, and the other is based on HNSW algorithm, with `L2` as the distance metric. We use `M = 50` and `ef_construction = 350` as build parameters for HNSW index.
In every test case we first generate a random vector, and then perform a simple use case. The test cases are the following:

1. Add a random new vector to the HNSW index
2. Delete a vector from the HNSW index
3. Run `Top_K` query over the flat index (using brute-force search), for `k=10`
4. Run `Top_K` query over the flat index (using brute-force search), for `k=100`
5. Run `Top_K` query over the flat index (using brute-force search), for `k=500`
6. Run `Top_K` query over the HNSW index, for `k=10`, using `ef_runtime=500`
7. Run `Top_K` query over the HNSW index, for `k=100`, using `ef_runtime=500`
8. Run `Top_K` query over the HNSW index, for `k=500`, using `ef_runtime=500`

#### Batch iterator benchmark

The purpose of this test suit is to benchmark the batched search feature. The batch iterator is a handle which enables running `Top_K` query in batches, by asking for the next best `n` results repeatedly, until there are no more results to return. We use for this test suit the same indices that were built for the "basic benchmark". The test cases are:

1. Run `Top_K` query over the flat index in batches of 10 results, until 1000 results are obtained.
2. Run `Top_K` query over the flat index in batches of 100 results, until 1000 results are obtained.
3. Run `Top_K` query over the flat index in batches of 100 results, until 10000 results are obtained. 
4. Run `Top_K` query over the flat index in batches of 1000 results, until 10000 results are obtained.
5. Run `Top_K` query over the HNSW index in batches of 10 results, until 1000 results are obtained.
6. Run `Top_K` query over the HNSW index in batches of 100 results, until 1000 results are obtained.
7. Run `Top_K` query over the HNSW index in batches of 100 results, until 10000 results are obtained, using `ef_runtime=500` in every batch.
8. Run `Top_K` query over the HNSW index in batches of 1000 results, until 10000 results are obtained, using `ef_runtime=500` in every batch.
9. Run regular `Top_K` query over the flat index for `k=1000`. 
10. Run regular `Top_K` query over the HNSW index for `k=1000`, using `ef_runtime=500`.
11. Run regular `Top_K` query over the flat index for `k=10000`.
12. Run regular `Top_K` query over the HNSW index for `k=10000`, using `ef_runtime=500`.


## ANN-Benchmark

[ANN-Benchmarks](http://ann-benchmarks.com/) is a benchmarking environment for approximate nearest neighbor algorithms search (for additional information, refer to the project's [github repository](https://github.com/erikbern/ann-benchmarks)).  Each algorithm is benchmarked on pre-generated commonly use datasets (in HDF5 formats).
The `bm_dataset.py` script uses some of ANN-Benchmark datasets to measure this library performance in the same manner. The following datasets are downloaded and benchmarked:

1. glove-25-angular
2. glove-50-angular
3. glove-100-angular
4. glove-200-angular
5. mnist-784-euclidean
6. sift-128-euclidean

For each dataset, the script will build an HNSW index with pre-defined build parameters, and persist it to a local file in `./data` directory that will be generated (index file name for example: `glove-25-angular-M=16-ef=100.hnsw`). Note that if the file already exists in this path, the entire index will be loaded instead of rebuilding it. Then, for 3 different pre-defined `ef_runtime` values, 1000 `Top_K` queries will be executed for `k=10` (these parameters can be modified easily in the script). For every configuration, the script outputs the following statistics:

- Average recall
- Query-per-second when running in brute-force mode
- Query-per-second when running with HNSW index

To reproduce this benchmark, first install the project's python bindings, and then invoke the script. From the project's root directory, you should run:
```py
python3 tests/benchmark/bm_datasets.py
```
