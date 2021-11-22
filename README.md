[![CircleCI](https://circleci.com/gh/RedisLabsModules/VectorSimilarity/tree/main.svg?style=svg)](https://circleci.com/gh/RedisLabsModules/VectorSimilarity/tree/main)

# VectorSimilarity

This repo exposes C API for using vector similarity search.
Allows Creating indices of vectors and searching for top K similar to some vector in two methods: brute force, and by using hnsw algorithm (probabilistic).

The API header files are `vec_sim.h` and `query_results.h`, which are located in `src/VecSim`.

## Build

```
./deps/readies/bin/getpy3
./sbin/system-setup.py
make
```

## Python bindings

Examples for using the python bindings to run vector similarity search can be found in `tests/flow`. 
To build the python wheel, first create a dedicated virtualenv using python 3.6 and higher. Then, activate the environment, install the dependencies and build the package. Please note, due to the way poetry generates a setup.py, you may have to erase it prior to re-running *poetry build*.

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
