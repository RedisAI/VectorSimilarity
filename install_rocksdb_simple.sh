#!/bin/bash

# Simple RocksDB Installation Script
# Quick installation without extensive error checking

set -e

# Default values
USE_ASAN=false
USE_UBSAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --asan)
            USE_ASAN=true
            shift
            ;;
        --ubsan)
            USE_UBSAN=true
            shift
            ;;
        --all-sanitizers)
            USE_ASAN=true
            USE_UBSAN=true
            shift
            ;;
        -h|--help)
            echo "Simple RocksDB Installation Script"
            echo
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --asan               Build with AddressSanitizer"
            echo "  --ubsan              Build with UndefinedBehaviorSanitizer"
            echo "  --all-sanitizers     Build with ASAN + UBSAN"
            echo "  -h, --help           Show this help"
            echo
            echo "Examples:"
            echo "  $0                           # Standard installation"
            echo "  $0 --asan                    # With AddressSanitizer"
            echo "  $0 --all-sanitizers          # With ASAN + UBSAN"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Installing RocksDB 10.5.1..."

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget \
    libgflags-dev libsnappy-dev zlib1g-dev libbz2-dev \
    liblz4-dev libzstd-dev libgtest-dev pkg-config

# Download and build
cd /tmp
wget https://github.com/facebook/rocksdb/archive/refs/tags/v10.5.1.tar.gz
tar -xzf v10.5.1.tar.gz
cd rocksdb-10.5.1

mkdir build && cd build

# Prepare sanitizer flags
SANITIZER_FLAGS=""
BUILD_TYPE="Release"

if [ "$USE_ASAN" = true ]; then
    SANITIZER_FLAGS="$SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer"
    BUILD_TYPE="Debug"
    echo "Building with AddressSanitizer (ASAN)"
fi

if [ "$USE_UBSAN" = true ]; then
    SANITIZER_FLAGS="$SANITIZER_FLAGS -fsanitize=undefined -fsanitize=integer"
    BUILD_TYPE="Debug"
    echo "Building with UndefinedBehaviorSanitizer (UBSAN)"
fi

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_FLAGS="-Wno-error=restrict $SANITIZER_FLAGS" \
    -DCMAKE_C_FLAGS="$SANITIZER_FLAGS" \
    -DWITH_GFLAGS=ON \
    -DWITH_SNAPPY=ON \
    -DWITH_LZ4=ON \
    -DWITH_ZSTD=ON \
    -DWITH_BZ2=ON \
    -DWITH_ZLIB=ON \
    -DWITH_TESTS=OFF \
    -DWITH_BENCHMARK_TOOLS=OFF \
    -DWITH_TOOLS=OFF \
    -DUSE_RTTI=ON \
    -DFAIL_ON_WARNINGS=OFF

make -j$(nproc)
sudo make install
sudo ldconfig

# Cleanup
cd /
rm -rf /tmp/rocksdb-10.5.1 /tmp/v10.5.1.tar.gz

echo "RocksDB installation completed!"
echo "Version: $(pkg-config --modversion rocksdb 2>/dev/null || echo "10.5.1")"
