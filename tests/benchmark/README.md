# Vector Similarity benchmarks

## Table of contents
* [Overview](#overview)
* [Run benchmarks](#run-benchmarks)
* [Available sets](#available-sets)  
    - [BM_VecSimBasics](#bm_vecsimbasics)  
    - [BM_BatchIterator](#bm_batchiterator)  
    - [BM_VecSimUpdatedIndex](#bm_vecsimupdatedindex)  
* [The benchmarks directory](#the-benchmarks-directory)
    - [How to define and register a new test](#how-to-define-and-register-a-new-test)
* [Fixtures](#fixtures)
    - [Benchmarks fixture classes hierarchy](#benchmarks-fixture-classes-hierarchy)
* [Cmake remarks](#cmake-remarks)
* [ann-benchmarks](#ann-benchmark)

# Overview
We use [Google benchmark](https://github.com/google/benchmark) tool to run micro-benchmarks for the vector indexes functionality.  
Google benchmark is a popular tool for benchmark code snippets, similar to unit tests. It allows a quick way of estimating the runtime of each test based on several (identical) runs, and print the results as output. For some tests, the output includes additional "Recall" metric, which indicates the accuracy in case of approximate search.  
**The recall** is calculated as the number of results returned by the approximate algorithm (HNSW in this case) which belong to the "ground truth" results (calculated by the brute force algorithm), divided by the ground truth results number. 

# Run benchmarks
## Required files
The serialized HNSW indices files that are used for micro-benchmarking and running ann-benchmark can be found in
`tests/benchmark/data/hnsw_indices.txt`.  
To download all the required files run from the repository root directory:
```asm
wget -q -i tests/benchmark/data/hnsw_indices.txt -P tests/benchmark/data
```
To run all test sets, call the following commands from the project root dir:
```asm
make benchmark
```
### Running a Subset of Benchmarks
To run only a subset of benchmarks that match a specified `<regex>`, set `BENCHMARK_FILTER=<regex>` environment variable. For example:  
```asm
make benchmark BENCHMARK_FILTER=fp32*
```

# Available sets
There are currently 3 sets of benchmarks available: `BM_VecSimBasics` , `BM_BatchIterator` and `BM_VecSimUpdatedIndex`. Each is templated according to the index data type.
## BM_VecSimBasics
For each combination of data type (fp32/fp64) and label type (single/multi) the following test cases are included:
1. Mesure index total `memory` (runtime is irrelevant for this use case, just the memory metric) 
2. `AddLabel` - adds `DEFAULT_BLOCK_SIZE (= 1024)` new labels to the index from the test_query_file. Note that for a single value index each label contains one vector, meaning that the number of new labels equals the number of new vectors.  
**results:** average time per label, average memory addition per vector and vectors per label.  
*At the end of the benchmark, we delete the added labels*
3. `DeleteLabel` - Deletes `DEFAULT_BLOCK_SIZE (= 1024)` labels from the index. Note that for a single value index each label contains one vector, meaning that the number of deleted labels equals the number of deleted vectors.  
**results:** average time per label, average memory addition per vector (a negative value means that the memory has decreased).  
*In each iteration, before deletion, we pause the time measurement to store the deleted label data so we can re-add it to the index at the end of the benchmark*
4. Run `Top_K` query over the flat index (using brute-force search), for each `k=10`, `k=100` and `k=500`  
**results:** average time per iteration
5. Run **100** iterations of `Top_K` query over the HNSW index, for each pair of `{ef_runtime, k}` from the following:  
    `{ef_runtime=10, k=10}`   
    `{ef_runtime=200, k=10}`   
    `{ef_runtime=100, k=100}`   
    `{ef_runtime=200, k=100}`   
    `{ef_runtime=500, k=500}`   
**results:** average time per iteration, Recall  
6. Run `Range` query over the flat index (using brute-force search), for each `radius=0.2`, `radius=0.35` and `radius=0.5`  
**results:** average time per iteration, average results number per iteration
7. Run **100** iterations of `Range` query over the HNSW index, for each pair of `{radius, epsilon}` from the following:  
    `{radius=0.2, epsilon=0.001}`   
    `{radius=0.2, epsilon=0.01}`    
    `{radius=0.2, epsilon=0.1}`     
    `{radius=0.35, epsilon=0.001}`   
    `{radius=0.35, epsilon=0.01}`      
    `{radius=0.35, epsilon=0.1}`      
    `{radius=0.5, epsilon=0.001}`   
    `{radius=0.5, epsilon=0.01}`   
    `{radius=0.5, epsilon=0.1}`   
**results:** average time per iteration, average results number per iteration, Recall  

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
2. Variable batch size - Run `Top_K` query where in each iteration the batch size is increased by factor of 2, for each pair of `{batch initial size, number of batches}` from the following:  
`{batch initial size=10, number of batches=2}`  
`{batch initial size=10, number of batches=4}`  
`{batch initial size=100, number of batches=2}`  
`{batch initial size=100, number of batches=4}`  
`{batch initial size=1000, number of batches=2}`  
`{batch initial size=1000, number of batches=4}`  
**Flat index results:** Time per iteration
**HNSW index results:** Time per iteration, Recall, memory delta per iteration
3. Batches to Adhoc BF - In each iteration we run `Top_K` with an increasing ` batch size` (initial size=10, increase factor=2) for `number of batches` and then switch to ad-hoc BF. We define `step` as the ratio between the index size to the number of vectors to go over in ad-hoc BF. The tests runs for each pair of `{step, number of batches}` from the following:  
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
For this use case, we create four indices, two for each algorithm (flat and HNSW). The first index per algorithm contains 500K vectors added to an empty index. The second index per algorithm also contains 500K vectors, but this time they were added by overriding the 500K vectors of the aforementioned indices. Currently, we only test this for FP32 single-value index.  
The test cases are:
1. Index `total memory` **before** updating
2. Index `total memory` **after** updating
3. Run `Top_K` query over the flat index **before** updating (using brute-force search), for each `k=10`, `k=100` and `k=500`  
4. Run `Top_K` query over the flat index **after** updating (using brute-force search), for each `k=10`, `k=100` and `k=500`  
**results:** average time per iteration
5. Run **100** iterations of `Top_K` query over the HNSW index **before** updating, for each pair of `{ef_runtime, k}` from the following:  
    `{ef_runtime=10, k=10}`   
    `{ef_runtime=200, k=10}`   
    `{ef_runtime=100, k=100}`   
    `{ef_runtime=200, k=100}`   
    `{ef_runtime=500, k=500}`   
6. Run **100** iterations of `Top_K` query over the HNSW index **after** updating, for each pair of `{ef_runtime, k}` from the following: 
    `{ef_runtime=10, k=10}`   
    `{ef_runtime=200, k=10}`   
    `{ef_runtime=100, k=100}`   
    `{ef_runtime=200, k=100}`   
    `{ef_runtime=500, k=500}`   
**results:** average time per iteration, Recall  
7. Run `Range` query over the flat index (using brute-force search), for each `radius=0.2`, `radius=0.35` and `radius=0.5`  
**results:** average time per iteration, average results number per iteration
8. Run **100** iterations of `Range` query over the HNSW index, for each pair of `{radius, epsilon` from the following:  
    `{radius=0.2, epsilon=0.001}`   
    `{radius=0.2, epsilon=0.01}`   
    `{radius=0.2, epsilon=0.1}`   
    `{radius=0.35, epsilon=0.001}`   
    `{radius=0.35, epsilon=0.01}`   
    `{radius=0.35, epsilon=0.1}`   
    `{radius=0.5, epsilon=0.001}`   
    `{radius=0.5, epsilon=0.01}`   
    `{radius=0.5, epsilon=0.1}`   
**results:** average time per iteration, average results number per iteration, Recall  

# The benchmarks directory 
This directory contains the following:
* **Header** files with the classes that are used to define the benchmark test routines. See [Fixtures](#fixtures) section for details.   
It also includes `bm_definitions.h` header that contains benchmarks utils such as:
    - `IndexType` struct
    - common typedefs
    - macro shortcuts to the static data members of `BM_VecSimGeneral` and `BM_VecSim_Index<index_type_t>`
* **subdirectory bm_initialization** - Each file is associated with a test set (basics/batch etc) of a particular **data type**, and fits for both single and multi indices. The purpose  of these files is to avoid code replications. Here we define and register the tests using Google benchmarks macros.  See [How to define and register a new test](#how-to-define-and-register-a-new-test) section.
* **subdirectory run_files** - contains a file for each combination of: set of benchmark tests, index data type, index type(single/multi).  
In this file we:
    - Initialize **BM_VecSimGeneral static data members** such as is_multi, dim, n_vectors, hnsw_index_file, etc, according to the set of parameters of the index loaded from the specified file.
    - In addition, the macro `BM_FUNC_NAME(bm_func, algo)` is defined to be used in the initialization file. The tests' names are used to generate the graphs on the Grafana dashboard. If two tests share the same name, they will appear on the same graph. Hence, This macro has to **define unique benchmark tests names**. (see [Tests naming](#tests-naming))
    - Include the bm_initialization file that calls google benchmark definitions and registrations macros.
    - Call `BENCHMARK_MAIN()` 


## How to define and register a new test
Google benchmarks library supports templated fixture (see [Google benchmarks user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md#templated-benchmarks) regarding this topic).  

### Definition:  
`BENCHMARK_TEMPLATE_DEFINE_F(fixture_name, test_name, data type)(benchmark::State &st){fixture_name::test_implementation(st, test_args);}`  
In the definition we:
1. Bind the test to a specific fixture
2. Give it a unique name (see [Tests naming](#tests-naming))
3. Specify the fixture data type (currently available: `fp32_index_t` and `fp64_index_t`) (see [Fixture template type](#fixture-template-type).  
4. Call the function, defined in the fixture class, that implements the required test case.
### Registration
`BENCHMARK_REGISTER_F(test_fixture, test_name)->Args({val1, val2})->ArgNames({"name1", "name2"})->Iterations(val0)->Unit(benchmark::kMillisecond)`  
Here we register to the benchmarks running loop the test that was bind to `test_fixture` and has `test_name` .  
*NOTE:* The **registration order** is important. Changing it might violate the ability to compare between one benchmark run to another. The reason to that is that we have tests (such as addLabel and DeleteLabel) that modify the hnsw graph.
# Fixtures
## Fixture template type
Since Google benchmarks supports fixture with only one template argument, we use a helper struct to define multiple types.  
```asm
// benchmarks/bm_definitions.h
template <VecSimType type, typename DataType, typename DistType = DataType>
struct IndexType {

    static VecSimType get_index_type() { return type; }

    typedef DataType data_t;
    typedef DistType dist_t;
};

// benchmarks/bm_vecsim_index.h
template <typename index_type_t>
class BM_VecSimIndex : public BM_VecSimGeneral {
    
    // Parsing the struct's types
    using data_t = typename index_type_t::data_t;
    using dist_t = typename index_type_t::dist_t;
    ...
    static std::vector<std::vector<data_t>> queries;

    // Using the VecSimType getter 
    static void Initialize() {
        VecSimType type = index_type_t::get_index_type();
        ...
    }
};

```
## Benchmarks fixture classes hierarchy 
You can find the hierarchy graph, defining the main members of each class here:  
[Benchmarks directory hierarchy graph](https://docs.google.com/presentation/d/1_vqZ50--H9Jv1yf-Pczq30n-kjtO9LW_XYOuH-D0Xjk/edit#slide=id.p).

### class BM_VecSimGeneral
Defined in bm_vecsim_general.h  
This class inherits from includes all the benchmark::Fixture.  
It includes the index parameters as static data members. The static data members are initialized in the .cpp files under benchmark/run_files.  
To refer a member from one of the derived classes, BM_VecSimGeneral:: must be used. Shortcut macros such as `#define N_QUERIES BM_VecSimGeneral::n_queries` are availble in bm_definitions.h.
### Remarks
* block_size is public because it is used to define the number of iterations on some test cases

### class BM_VecSim_Index<index_type_t> : public BM_VecSimGeneral
Defined in bm_vecsim_index.h  
This class purpose is to initialize the indices and queries data.
It contains:
* A vector `indices` of `VecSimIndex *` type. One can refer to an index using `VecSimAlgo` enum, where `VecSimAlgo = VecSimAlgo_BF = 0` and `VecSimAlgo = VecSimAlgo_HNSWLIB = 1`.   
First, We load a serialized hnsw index from a file, read this index vectors one by one and add the vectors manually to the flat index.  
*NOTE:* The updated tests set is using this vector to save its 4 indices as described in [BM_VecSimUpdatedIndex](#bm_vecsimupdatedindex), where `indices[0]` and `indices[1]` are the BF and HNSW **before** the update, respectively, and `indices[2]` and `indices[3]` are the BF and HNSW **after** the update, respectively.
* A vector `queries` contains query vectors of `data_t` type.  
Both Vectors are populated using the files defined in `BM_VecSimGeneral`.
* Reference count - each benchmark definition instantiate a new object of the `test_fixture`. To avoid calling the initialization function multiple times, we use the reference count in the constructor- if it's 0 we initialize the vectors mentioned above, otherwise just increasing the reference count.  
Using the reference count we also make sure we don't free the indices before we finished all the set's test - the class destructor decreases the reference count, and if it have reached 0 - we free the indices.
### Remarks
* `indices` and `queries` vectors needs to be explicitly initialized for a specific type. For example:
```asm
template <>
std::vector<std::vector<float>> BM_VecSimIndex<fp32_index_t>::queries{};

template <>
std::vector<VecSimIndex *> BM_VecSimIndex<fp32_index_t>::indices{};
```
### class BM_VecSimCommon<index_type_t> : public BM_VecSimIndex<index_type_t>
Defined in bm_common.h  
In this class we define the test routines common to the **basics** benchmark and the [**updated index**](#bm_vecsimupdatedindex) benchmarks:
* `TopK_BF`
* `TopK_HNSW`
* `Memory_FLAT` 
* `Memory_HNSW`  
### Remarks
These function has `index_offset` parameter (defaults to 0) to be used when we want to apply the function on an updated index.  
In addition, in this file we define macros with the arguments for the `TopK` registration.

### class BM_VecSimBasics<index_type_t> : public BM_VecSimCommon<index_type_t>
Defined in bm_vecsim_basics.h  
In this class we define additional tests for the basic tests set:
* `AddLabel` - Add one label per iteration (meaning 1 vector for single value index, and multiple vector for multi value index)
* `DeleteLabel<algo_t>` - Delete one label per iteration (meaning 1 vector for single value index, and multiple vector for multi value index).  
`algo_t` is a specific index type (including multi/single) because we call `GetDataByLabel()`, which is only known to HNSW_Single/Multi and BruteForce_Single/Multi. This why this test case is defined separately in each .cpp file using `DEFINE_DELETE_LABEL` macro (defined in bm_vecsim_basics.h). The registration is done like all other test cases in the initialization file.  
* `Range_BF` 
* `Range_HNSW`  
In addition, in this file we define macros with the arguments for the `Range`, `AddLabel` and `DeleteLabel` registration.
### Remarks
The [basic benchmark set](#bm_vecsimbasics) includes functions defined both in `BM_VecSimCommon` and `BM_VecSimBasics`. Pay attention to refer the correct class when defining the test in the [initialization file](#how-to-define-and-register-a-new-test).
### class BM_VecSimUpdatedIndex<index_type_t> : public BM_VecSimCommon<index_type_t>
Defined in bm_updated_index.h  
In this class constructor we add the additional indices, the "updated" indices. The "before" indices are initialized in `BM_VecSimIndex`.  
This class defines no new test routines and uses only tests defined in `BM_VecSimCommon`.  
### Remarks
The constructor is called after we already defined the tests resides in `BM_VecSimCommon`, so `ref_count` is not zero and we cant rely on it to decide weather we should initialize the indices or not. This is why we have to hold another the `is_initialized` flag.  
Also, we keep the value of `ref_count` at the moment of initialization in `first_updatedBM_ref_count` to free the indices when `ref_count` decreased to this value.
### class BM_BatchIterator<index_type_t> : public BM_VecSimIndex<index_type_t>
Defined in bm_batch_iterator.h  
Here we implement the tests routines of [batch iterator set](#bm_batchiterator) to be called by the test definition in the [initialization file](#how-to-define-and-register-a-new-test).  
In this file we also define the arguments for each test case in the batch iterator tests set.

### Tests counters
* Tests counters finale values are printed in the benchmarks output next to the test results. 
* They can also be used as input data in the Grafana dashboard.
### Tests naming:
A test name will look something like this:  
`test_fixture<fixture_data_type>/test_name/arg1:val/iterations:val`  
for example:  
`BM_VecSimBasics<fp32_index_t>/Range_HNSW_Single/radiusX100:20/epsilonX1000:1/iterations:100`  
So test's name components are:
1. The test's fixture and the fixture type (if it's templated)
2. The name as given at definition and registration
3. arguments names and values  
*NOTE:* the fixture type (fp32_index_t, for example) is part of the name, so tests that use different fixture **or** the same fixture of a different type, **can** share the same name.  


# Cmake remarks
Each test binary includes only one file .cpp from benchmark/run_files because
1. We don't want to override BM_VecSimGeneral static data members
2. Memory (RAM) limitations.

## benchmarks.sh
In this file we define the names of the benchmarks binary files, **without** the `bm_` prefix. The names are used in:
1. cmake of the benchmarks - to define the executable file name (bm_${benchmark}), refer the .cpp files, 
2. Makefile - to execute the benchmarks (with the names defined the cmake) and define the results files names (${benchmark}_results).
3. benchmark.yml - read the results from the files according to the name defined in the Makefile.

# ANN-Benchmark

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