
# 1. Create build directory
mkdir -p bin/Linux-x86_64-debug-asan
cd bin/Linux-x86_64-debug-asan

# 2. Configure with CMake (includes ASAN)
CC=clang CXX=clang++ cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DVECSIM_BUILD_TESTS=ON \
  -DUSE_SVS=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DCMAKE_MODULE_PATH=/usr/local/lib/cmake/rocksdb \
  -DROCKSDB_ROOT_DIR=/usr/local \
  -DROCKSDB_INCLUDE_DIR=/usr/local/include \
  -DROCKSDB_LIBRARY=/usr/local/lib/librocksdb.so \
  -DCMAKE_EXE_LINKER_FLAGS="-lzstd -lsnappy -llz4 -lbz2 -lz" \
  -DCMAKE_SHARED_LINKER_FLAGS="-lzstd -lsnappy -llz4 -lbz2 -lz" \
  /home/ben/Repos/VectorSimilarity

# 3. Build
make test_hnsw_disk -j4

# 4. Run tests with ASAN
LD_PRELOAD=/lib/x86_64-linux-gnu/libasan.so.8 \
ASAN_OPTIONS=abort_on_error=0:detect_leaks=1 \
./unit_tests/test_hnsw_disk