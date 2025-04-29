[![nightly](https://github.com/RedisAI/VectorSimilarity/actions/workflows/event-nightly.yml/badge.svg)](https://github.com/RedisAI/VectorSimilarity/actions/workflows/event-nightly.yml)
[![codecov](https://codecov.io/gh/RedisAI/VectorSimilarity/branch/main/graph/badge.svg)](https://codecov.io/gh/RedisAI/VectorSimilarity)
[![CodeQL](https://github.com/RedisAI/VectorSimilarity/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/RedisAI/VectorSimilarity/actions/workflows/codeql-analysis.yml)
[![Known Vulnerabilities](https://snyk.io/test/github/RedisAI/VectorSimilarity/badge.svg)](https://snyk.io/test/github/RedisAI/VectorSimilarity)


# VectorSimilarity
Starting with version 8.0, RediSearch and this vector similarity library is an integral part of Redis. See https://github.com/redis/redis 

This repo exposes C API for using vector similarity search.
Allows Creating indices of vectors and searching for top K similar to some vector in two methods: brute force, and by using the hnsw algorithm (probabilistic).

The API header files are `vec_sim.h` and `query_results.h`, which are located in `src/VecSim`.

## Algorithms

All of the algorithms in this library are designed to work inside RediSearch and support the following features:
1. In place insert, delete, and update vectors in the index.
2. KNN queries - results can be ordered by score or ID.
3. Iterator interface for consecutive KNN queries.
4. Range queries
5. Multiple vector indexing for the same label (multi-value indexing)
6. 3rd party allocators

#### Datatypes SIMD support

| Operation | x86_64 | arm64v8 | Apple silicone |
|-----------|--------|---------|-----------------|
| FP32 Internal product |SSE, AVX, AVX512 | No SIMD support | No SIMD support |
| FP32 L2 distance |SSE, AVX, AVX512| No SIMD support | No SIMD support |
| FP64 Internal product |SSE, AVX, AVX512 | No SIMD support | No SIMD support |
| FP64 L2 distance |SSE, AVX, AVX512 | No SIMD support | No SIMD support |

### Flat (Brute Force)

Brute force comparison of the query vector `q` with the stored vectors. Vectors are stored in vector blocks, which are contiguous memory blocks, with configurable size.


### HNSW
Modified implementation of [hnswlib](https://github.com/nmslib/hnswlib). Modified to accommodate the above feature set.

## Build
For building you will need:
1. Python 3 as `python` (either by creating a virtual environment or setting your system python to point to the right python distribution)
2. gcc >= 10
3. cmake version >= 3.10

To build the main library, unit tests, and Python bindings in one command run
```
make
```

## Unit tests
To execute unit tests run

```
make unit_test
```
## Memory check

To run the unit tests with Valgrind run
```
make unit_test VALGRIND=1
```

## Python bindings

Examples of using the Python bindings to run vector similarity search can be found in `tests/flow`.
To build the Python wheel, first create a dedicated virtualenv using Python 3.7 and higher. Then, activate the environment, install the dependencies, and build the package. Please note, due to the way poetry generates a setup.py, you may have to erase it before re-running *poetry build*.

```
python -m venv venv
source venv/bin/activate
pip install poetry
poetry install
poetry build
```

To run in debug mode, replace the last two lines with:

```
DEBUG=1 poetry install
DEBUG=1 poetry build
```

After building the wheel, if you want to use the package you built, you will need to manually execute a *pip install dist/<package>.whl*. Remember to replace <package> with the complete package name.

### Testing Python bindings
This will create a new virtual environment (if needed), install the wheel, and execute the Python bindings tests
```
poetry run pytest tests/flow
```
Or you can use the make command:
```
make flow_test
```

# Benchmark

To benchmark the capabilities of this library, follow the instructions in the [benchmarks user guide](docs/benchmarks.md).
If you'd like to create your own benchmarks, you can find more information in the [developer guide](docs/benchmarks_developer.md).
