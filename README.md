[![nightly](https://github.com/RedisAI/VectorSimilarity/actions/workflows/nightly.yml/badge.svg)](https://github.com/RedisAI/VectorSimilarity/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/RedisAI/VectorSimilarity/branch/main/graph/badge.svg)](https://codecov.io/gh/RedisAI/VectorSimilarity)
[![CodeQL](https://github.com/RedisAI/VectorSimilarity/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/RedisAI/VectorSimilarity/actions/workflows/codeql-analysis.yml)
[![Known Vulnerabilities](https://snyk.io/test/github/RedisAI/VectorSimilarity/badge.svg)](https://snyk.io/test/github/RedisAI/VectorSimilarity)


# VectorSimilarity

This repo exposes C API for using vector similarity search.
Allows Creating indices of vectors and searching for top K similar to some vector in two methods: brute force, and by using hnsw algorithm (probabilistic).

The API header files are `vec_sim.h` and `query_results.h`, which are located in `src/VecSim`.

## Algorithms

### Flat (Brute Force)
TBD

### HNSW
TBD

## Build
For building you will need:
1. Python 3 as `python` (either by creating virtual environment or setting your system python to point to the right python distribution)
2. gcc >= 10
3. cmake version >= 3.10

To build the main library, unit tests and python byndings in one command run
```
make
```

## Unit tests
To execute unit tests run

```
make unit_test
```
## Memory check

To run unit with valgrind run
```
make unit_test VALGRIND=1
```

## Python bindings

Examples for using the python bindings to run vector similarity search can be found in `tests/flow`.
To build the python wheel, first create a dedicated virtualenv using python 3.7 and higher. Then, activate the environment, install the dependencies and build the package. Please note, due to the way poetry generates a setup.py, you may have to erase it prior to re-running *poetry build*.

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

### Testing python bindings
This will create a new tox environment, install the wheel and execute the python bindings tests
```
tox -e flowenv
```

# Benchmark

To benchmark the capabilities of this library, follow the instructions in the [benchmark docs section](docs/benchmarks.md).
