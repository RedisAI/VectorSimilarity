# VecSim ARM baseline build

The baseline was built on `dorer-arm-big` in `/home/ubuntu/pipnn-poc-20261114`, at detached commit `54a841ae841828d53a482a826d6ace7753bc5450`. The commands below are run on that host unless noted otherwise. The build used GCC 12.3.0, CMake `Release`, tests and test helpers disabled, and SVS disabled. CMake selected the ARM SIMD configurations supported by the host.

Create the worktree from the existing local clone and initialize the pinned SVS submodule from local objects:

```sh
mkdir -p /home/ubuntu/pipnn-poc-20261114
git -C /home/ubuntu/wt/arm-pr6 worktree add --detach /home/ubuntu/pipnn-poc-20261114/vecsim 54a841ae841828d53a482a826d6ace7753bc5450
git -C /home/ubuntu/pipnn-poc-20261114/vecsim submodule update --init --no-fetch deps/ScalableVectorSearch
```

The pinned revision's SVS-disabled build requires the minimal guard patch saved at `/home/ubuntu/pipnn-poc-20261114/vecsim-svs-off.patch`. It guards SVS includes and shared-pool calls; it does not modify HNSW or SIMD code. Apply it to a fresh worktree before building:

```sh
git -C /home/ubuntu/pipnn-poc-20261114/vecsim apply /home/ubuntu/pipnn-poc-20261114/vecsim-svs-off.patch
```

Configure and build the library, with at most four jobs:

```sh
cmake -S /home/ubuntu/pipnn-poc-20261114/vecsim -B /home/ubuntu/pipnn-poc-20261114/vecsim-build -DCMAKE_BUILD_TYPE=Release -DVECSIM_BUILD_TESTS=OFF -DVECSIM_BUILD_TEST_HELPERS=OFF -DUSE_SVS=OFF
cmake --build /home/ubuntu/pipnn-poc-20261114/vecsim-build --target VectorSimilarity -j4 > /home/ubuntu/pipnn-poc-20261114/vecsim-build.log 2>&1
```

The host approval marker was supplied as a leading prefix to the local SSH command: `VECSIM_BUILD_MACHINE_CONFIRMED=1 ssh dorer-arm-big '…'`. The source is `vecsim_bench.cpp` in this directory and was copied to the remote root before compilation. Link the driver against the exact library dependencies:

```sh
cd /home/ubuntu/pipnn-poc-20261114
g++ -O3 -std=c++20 -Ivecsim/src vecsim_bench.cpp -Wl,--start-group vecsim-build/src/VecSim/libVectorSimilarity.a vecsim-build/src/VecSim/spaces/libVectorSimilaritySpaces.a vecsim-build/src/VecSim/spaces/libVectorSimilaritySpaces_no_optimization.a vecsim-build/_deps/cpu_features-build/libcpu_features.a -Wl,--end-group -lpthread -ldl -o vecsim_bench
```

Copy `smoke_data.py` from this directory to the remote root, then generate the small brute-force reference and run the smoke test:

```sh
cd /home/ubuntu/pipnn-poc-20261114
python3 smoke_data.py smoke
./vecsim_bench smoke/base.fbin smoke/query.fbin smoke/truth.ibin 16 200 1 200
```

The smoke run used 200 base vectors and 12 queries, with dimension 16, and reported recall@10 of 1.0. Its timings are not benchmark results. The C API uses the HNSW constructor's fixed seed of 100; the driver's `repeat` argument is a run identifier, not a seed.
