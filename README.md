[![CircleCI](https://circleci.com/gh/RedisLabsModules/VectorSimilarity/tree/main.svg?style=svg)](https://circleci.com/gh/RedisLabsModules/VectorSimilarity/tree/main)

# VectorSimilarity

This repo exposes C API for using hnswlib implemetation for vector similarity search.
Allows Creating indices of vectors and searching for most K similar to some vector in two methods: brute force, and by using hnsw algorithm (probablistic).

The API headres is available in hnsw_c.h file.

## Build

```
cd opt
./build.sh
```
