[![CircleCI](https://circleci.com/gh/RedisLabsModules/VectorSimilarity/tree/main.svg?style=svg)](https://circleci.com/gh/RedisLabsModules/VectorSimilarity/tree/main)

# VectorSimilarity

This repo exposes C API for using hnswlib implemetation for vector similarity search.
Allows Creating indices of vectors and searching for most K similar to some vector in two methods: brute force, and by using hnsw algorithm (probablistic).

The API headres is available in hnsw_c.h file.

## Build

```
./deps/readies/bin/getpy3
./sbin/system-setup.py
make
```

## Building python wheel

Create a dedicated virtualenv using python 3.6 and higher. Activate the environment, install the dependencies and build the package. Please note, due to the way poetry generates a setup.py, you may have to erase it prior to re-running *poetry build*.

```
python -m venv venv
source venv/bin/activate
pip install poetry
poetry install
poetry build
```

After building the wheel, if you want to use the package you built, you will need to manually execute a *pip install dist/<package>.whl*. Remember to replace <package> with the complete package name.
