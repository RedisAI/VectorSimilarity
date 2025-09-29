# RocksDB Installation Scripts

This directory contains scripts to install RocksDB from source with C++20 support and all necessary compression libraries.

## Scripts Available

### 1. `install_rocksdb.sh` - Full Installation Script
A comprehensive script with error checking, logging, and verification.

**Features:**
- ✅ Dependency checking and installation
- ✅ Colored output and progress logging
- ✅ Installation verification
- ✅ Automatic cleanup
- ✅ Help and version options
- ✅ Error handling and rollback

**Usage:**
```bash
# Show help
./install_rocksdb.sh --help

# Show version
./install_rocksdb.sh --version

# Standard installation
./install_rocksdb.sh

# With AddressSanitizer (for memory debugging)
./install_rocksdb.sh --asan

# With all common sanitizers
./install_rocksdb.sh --all-sanitizers

# Custom sanitizer combination
./install_rocksdb.sh --asan --ubsan
```

### 2. `install_rocksdb_simple.sh` - Quick Installation
A minimal script for quick installation without extensive error checking.

**Usage:**
```bash
# Standard installation
./install_rocksdb_simple.sh

# With AddressSanitizer
./install_rocksdb_simple.sh --asan

# With all sanitizers
./install_rocksdb_simple.sh --all-sanitizers

# Show help
./install_rocksdb_simple.sh --help
```

### 3. `build_with_sanitizers.sh` - VectorSimilarity Build Script
A comprehensive script for building VectorSimilarity with sanitizer support.

**Features:**
- ✅ Multiple sanitizer options (ASAN, MSAN, TSAN, UBSAN)
- ✅ Automatic test execution
- ✅ Clean build option
- ✅ Custom build directory support
- ✅ Colored output and progress logging

**Usage:**
```bash
# Standard debug build
./build_with_sanitizers.sh

# With AddressSanitizer
./build_with_sanitizers.sh --asan

# With all sanitizers
./build_with_sanitizers.sh --all-sanitizers

# Clean build with ASAN
./build_with_sanitizers.sh --asan --clean

# Custom build directory
./build_with_sanitizers.sh --build-dir custom_build

# Show help
./build_with_sanitizers.sh --help
```

## What Gets Installed

- **RocksDB Version:** 10.5.1
- **Install Location:** `/usr/local/`
- **Libraries:** `librocksdb.so`, compression libraries
- **Headers:** `/usr/local/include/rocksdb/`
- **CMake Support:** `/usr/local/lib/cmake/rocksdb/`

## Dependencies Installed

The scripts automatically install these system packages:
- `build-essential` - Compiler toolchain
- `cmake` - Build system
- `git` - Version control
- `wget` - Download tool
- `libgflags-dev` - Command line flags
- `libsnappy-dev` - Snappy compression
- `zlib1g-dev` - Zlib compression
- `libbz2-dev` - Bzip2 compression
- `liblz4-dev` - LZ4 compression
- `libzstd-dev` - Zstandard compression
- `libgtest-dev` - Google Test framework
- `pkg-config` - Package configuration

## Build Configuration

The scripts configure RocksDB with:
- **C++ Standard:** C++20
- **Build Type:** Release (Debug when sanitizers are enabled)
- **Compression:** All supported formats (Snappy, LZ4, ZSTD, BZ2, Zlib)
- **Features:** RTTI enabled, warnings as errors disabled
- **Tools:** Tests and benchmarks disabled for smaller footprint

## Sanitizer Support

The scripts support building RocksDB with various sanitizers for debugging:

### Available Sanitizers

| Sanitizer | Flag | Purpose | Performance Impact |
|-----------|------|---------|-------------------|
| **AddressSanitizer** | `--asan` | Detects memory errors (buffer overflows, use-after-free) | ~2x slower |
| **MemorySanitizer** | `--msan` | Detects uninitialized memory reads | ~3x slower |
| **ThreadSanitizer** | `--tsan` | Detects data races in multi-threaded programs | ~5-15x slower |
| **UndefinedBehaviorSanitizer** | `--ubsan` | Detects undefined behavior (integer overflow, etc.) | ~1.5x slower |

### Common Combinations

```bash
# Most common for memory debugging
./install_rocksdb.sh --all-sanitizers

# Memory debugging only
./install_rocksdb.sh --asan

# Thread safety debugging
./install_rocksdb.sh --tsan

# All sanitizers (comprehensive debugging)
./install_rocksdb.sh --asan --msan --tsan --ubsan
```

### Performance Considerations

- **Debug builds** are automatically used when sanitizers are enabled
- **Memory usage** increases significantly with sanitizers
- **Compilation time** increases due to additional instrumentation
- **Runtime performance** is significantly slower (see table above)

### When to Use Sanitizers

- **Development:** Use `--asan` for most memory debugging
- **Testing:** Use `--all-sanitizers` for comprehensive testing
- **Production:** Never use sanitizers in production builds
- **CI/CD:** Use sanitizers in automated testing pipelines

## Verification

After installation, you can verify RocksDB is working:

```bash
# Check library version
pkg-config --modversion rocksdb

# Test compilation
echo '#include <rocksdb/version.h>
#include <iostream>
int main() { 
    std::cout << "RocksDB Version: " << ROCKSDB_MAJOR << "." << ROCKSDB_MINOR << "." << ROCKSDB_PATCH << std::endl; 
    return 0; 
}' > test_rocksdb.cpp

g++ -std=c++20 -I/usr/local/include -L/usr/local/lib -lrocksdb test_rocksdb.cpp -o test_rocksdb
./test_rocksdb
rm test_rocksdb.cpp test_rocksdb
```

## Complete Workflow

### Option 1: Using the Build Script (Recommended)

```bash
# 1. Install RocksDB with sanitizers
./install_rocksdb.sh --all-sanitizers

# 2. Build VectorSimilarity with sanitizers
./build_with_sanitizers.sh --all-sanitizers

# 3. Run tests
cd bin/Linux-x86_64-debug-asan
ASAN_OPTIONS=abort_on_error=0:detect_leaks=1 ./unit_tests/test_hnsw_disk
```

### Option 2: Manual Build

```bash
# 1. Install RocksDB
./install_rocksdb.sh --asan

# 2. Create build directory
mkdir -p bin/Linux-x86_64-debug-asan
cd bin/Linux-x86_64-debug-asan

# 3. Configure with CMake
CC=clang CXX=clang++ cmake \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DUSE_ASAN=ON \
  -DVECSIM_BUILD_TESTS=ON \
  -DUSE_SVS=OFF \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -DCMAKE_MODULE_PATH=/usr/local/lib/cmake/rocksdb \
  /path/to/VectorSimilarity

# 4. Build
make test_hnsw_disk

# 5. Run tests
ASAN_OPTIONS=abort_on_error=0:detect_leaks=1 ./unit_tests/test_hnsw_disk
```

### Option 3: Quick Setup

```bash
# Install and build everything with sanitizers
./install_rocksdb_simple.sh --all-sanitizers
./build_with_sanitizers.sh --all-sanitizers --clean
```

## Troubleshooting

### Permission Issues
If you get permission errors, make sure you have sudo access:
```bash
sudo -v
```

### Library Not Found
If you get "library not found" errors, update the library cache:
```bash
sudo ldconfig
```

### CMake Not Finding RocksDB
Make sure the CMake module path is set:
```bash
export CMAKE_MODULE_PATH=/usr/local/lib/cmake/rocksdb:$CMAKE_MODULE_PATH
```

### Compilation Errors
If you get C++20 compilation errors, ensure you're using a recent compiler:
```bash
g++ --version  # Should be 9.0+ for C++20 support
clang++ --version  # Should be 10.0+ for C++20 support
```

## Uninstalling

To remove RocksDB:
```bash
sudo rm -rf /usr/local/lib/librocksdb*
sudo rm -rf /usr/local/include/rocksdb
sudo rm -rf /usr/local/lib/cmake/rocksdb
sudo ldconfig
```

## Notes

- The scripts require internet access to download RocksDB source
- Installation takes 5-15 minutes depending on your system
- The build uses all available CPU cores for faster compilation
- All temporary files are automatically cleaned up
