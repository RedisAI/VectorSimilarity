# PiPNN ARM build

This records the successful proof-of-concept build on `dorer-arm-big`, using GCC 12.3.0 and CMake Release (`-O3 -DNDEBUG -march=native -ffast-math`). The upstream revision and resolved submodule and FetchContent revisions are in [metadata.txt](metadata.txt). FetchContent uses moving branches for several dependencies; a fresh configure may therefore resolve different revisions unless the recorded revisions are supplied separately.

## Fresh checkout and build

Run on the ARM host, with the patch and `pipnn_query.cpp` copied from this directory. Replace `ARTIFACT_DIR` and `WORK_DIR` with their absolute paths. The top-level submodules are initialized without `--recursive`: the MIRAGE checkout has a stale nested submodule entry that makes recursive initialization fail.

```sh
git clone https://github.com/ParAlg/PiPNN.git "$WORK_DIR"
cd "$WORK_DIR"
git checkout 443a328dc71a3b7be753d52ca2b3c9a4ba77f3fd
git submodule update --init
git apply "$ARTIFACT_DIR/arm-port.patch"
cp "$ARTIFACT_DIR/pipnn_query.cpp" algorithms/PipNN/pipnn_query.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target neighbors-pipnn_FLOAT_T_EUCLIDEAN -j4
cmake --build build --target pipnn_query -j4
```

The patch combines three changes: ARM-compatible headers and aligned allocation; a PiPNN-only CMake configuration; and a CMake target linking the separate query driver to Parlay, Eigen, Abseil and Highway. The build restriction is not an upstream requirement or part of the algorithm: it skips unrelated targets, including MIRAGE, whose configuration otherwise failed on this host because BLAS was unavailable. The query source is a separate file and must be copied as shown. Eigen and Highway remain in the PiPNN build.

Configuration fetches dependencies from the network, including Eigen, Parlay, Abseil, Highway and Google Benchmark. The top-level submodule checkout of Parlay is distinct from the version actually fetched by CMake; use the FetchContent revisions in the metadata to identify the tested build.

The resulting executables are `build/algorithms/PipNN/neighbors-pipnn_FLOAT_T_EUCLIDEAN` and `build/algorithms/PipNN/pipnn_query`. Set `PARLAY_NUM_THREADS` at runtime. The query driver accepts a graph file, base fbin, query fbin, ground-truth file, k, and one or more beam sizes:

```sh
PARLAY_NUM_THREADS=1 build/algorithms/PipNN/pipnn_query \
  GRAPH BASE.fbin QUERY.fbin TRUTH.bin 10 10 32 64 100
```

Ground truth contains two 32-bit integers (`nq`, `k`), then `nq*k` row-major 32-bit IDs, then `nq*k` row-major float32 distances. The driver warms the first 100 queries per beam, measures each query separately, and checks result count, uniqueness and bounds before reporting recall and latency percentiles. Its index-process RSS snapshot is taken after loading the graph and base vectors and before loading queries and ground truth; file sizes are reported separately.

## Checks performed

- ARM scalar L2, inner-product and norm fallbacks matched a double reference for dimensions 1 through 100 using the recorded release flags. The test is in `distance_check.cpp`; its output is in `distance_check.log`.
- The builder produced a graph for 512 random 8-dimensional vectors using EigenKNN and two workers.
- The query driver returned recall@10 of 1.0 on 20 independent queries against a 128-point, 13-dimensional fully connected graph, with beam 128 and independently computed brute-force ground truth. Its uniqueness and bounds checks passed. This check did not independently verify result ordering.

The remote proof-of-concept directory contains the configure, build and smoke logs. These checks establish basic portability and query correctness for the cases tested, not benchmark parity with the paper.
