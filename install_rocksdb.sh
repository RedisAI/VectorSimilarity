#!/bin/bash

# RocksDB Installation Script
# This script downloads, builds, and installs RocksDB from source
# with C++20 support and all necessary compression libraries

set -e  # Exit on any error

# Configuration
ROCKSDB_VERSION="10.5.1"
INSTALL_PREFIX="/usr/local"
BUILD_DIR="/tmp/rocksdb_build"
NUM_CORES=$(nproc)

# Sanitizer options
USE_ASAN=false
USE_MSAN=false
USE_TSAN=false
USE_UBSAN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This is not recommended for security reasons."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for required packages
    local required_packages=(
        "build-essential"
        "cmake"
        "git"
        "wget"
        "libgflags-dev"
        "libsnappy-dev"
        "zlib1g-dev"
        "libbz2-dev"
        "liblz4-dev"
        "libzstd-dev"
        "libgtest-dev"
        "pkg-config"
    )
    
    for package in "${required_packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            missing_deps+=("$package")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Installing missing dependencies..."
        sudo apt update
        sudo apt install -y "${missing_deps[@]}"
    fi
    
    log_success "All dependencies are installed"
}

# Download RocksDB source
download_rocksdb() {
    log_info "Downloading RocksDB ${ROCKSDB_VERSION}..."
    
    if [ -d "$BUILD_DIR" ]; then
        log_info "Removing existing build directory..."
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Download source tarball
    local rocksdb_url="https://github.com/facebook/rocksdb/archive/refs/tags/v${ROCKSDB_VERSION}.tar.gz"
    log_info "Downloading from: $rocksdb_url"
    
    if ! wget -q --show-progress "$rocksdb_url"; then
        log_error "Failed to download RocksDB source"
        exit 1
    fi
    
    # Extract source
    log_info "Extracting source..."
    tar -xzf "v${ROCKSDB_VERSION}.tar.gz"
    cd "rocksdb-${ROCKSDB_VERSION}"
    
    log_success "RocksDB source downloaded and extracted"
}

# Build and install RocksDB
build_rocksdb() {
    log_info "Building RocksDB with C++20 support..."
    
    cd "$BUILD_DIR/rocksdb-${ROCKSDB_VERSION}"
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Prepare sanitizer flags
    local sanitizer_flags=""
    local build_type="Release"
    
    if [ "$USE_ASAN" = true ]; then
        sanitizer_flags="$sanitizer_flags -fsanitize=address -fno-omit-frame-pointer"
        build_type="Debug"
        log_info "Building with AddressSanitizer (ASAN)"
    fi
    
    if [ "$USE_MSAN" = true ]; then
        sanitizer_flags="$sanitizer_flags -fsanitize=memory -fno-omit-frame-pointer"
        build_type="Debug"
        log_info "Building with MemorySanitizer (MSAN)"
    fi
    
    if [ "$USE_TSAN" = true ]; then
        sanitizer_flags="$sanitizer_flags -fsanitize=thread"
        build_type="Debug"
        log_info "Building with ThreadSanitizer (TSAN)"
    fi
    
    if [ "$USE_UBSAN" = true ]; then
        sanitizer_flags="$sanitizer_flags -fsanitize=undefined -fsanitize=integer"
        build_type="Debug"
        log_info "Building with UndefinedBehaviorSanitizer (UBSAN)"
    fi
    
    # Configure with CMake
    log_info "Configuring with CMake (Build type: $build_type)..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_FLAGS="-Wno-error=restrict $sanitizer_flags" \
        -DCMAKE_C_FLAGS="$sanitizer_flags" \
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
    
    # Build
    log_info "Building RocksDB (using $NUM_CORES cores)..."
    make -j"$NUM_CORES"
    
    # Install
    log_info "Installing RocksDB to $INSTALL_PREFIX..."
    sudo make install
    
    # Update library cache
    sudo ldconfig
    
    log_success "RocksDB built and installed successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check if libraries are installed
    local lib_path="$INSTALL_PREFIX/lib"
    local include_path="$INSTALL_PREFIX/include"
    
    if [ ! -f "$lib_path/librocksdb.so" ]; then
        log_error "RocksDB library not found at $lib_path/librocksdb.so"
        return 1
    fi
    
    if [ ! -d "$include_path/rocksdb" ]; then
        log_error "RocksDB headers not found at $include_path/rocksdb"
        return 1
    fi
    
    # Test compilation
    log_info "Testing compilation..."
    local test_file="/tmp/rocksdb_test.cpp"
    cat > "$test_file" << 'EOF'
#include <rocksdb/version.h>
#include <rocksdb/db.h>
#include <iostream>

int main() {
    std::cout << "RocksDB Version: " << ROCKSDB_MAJOR << "." << ROCKSDB_MINOR << "." << ROCKSDB_PATCH << std::endl;
    
    rocksdb::Options options;
    options.create_if_missing = true;
    
    rocksdb::DB* db;
    rocksdb::Status status = rocksdb::DB::Open(options, "/tmp/test_db", &db);
    
    if (status.ok()) {
        std::cout << "RocksDB test: SUCCESS" << std::endl;
        delete db;
    } else {
        std::cout << "RocksDB test: FAILED - " << status.ToString() << std::endl;
        return 1;
    }
    
    return 0;
}
EOF
    
    if g++ -std=c++20 -I"$include_path" -L"$lib_path" -lrocksdb "$test_file" -o /tmp/rocksdb_test; then
        if /tmp/rocksdb_test; then
            log_success "RocksDB installation verified successfully"
        else
            log_error "RocksDB test failed"
            return 1
        fi
    else
        log_error "Failed to compile RocksDB test"
        return 1
    fi
    
    # Cleanup test files
    rm -f "$test_file" /tmp/rocksdb_test
    rm -rf /tmp/test_db
    
    return 0
}

# Cleanup function
cleanup() {
    log_info "Cleaning up build directory..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
    fi
}

# Main function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    log_info "Starting RocksDB installation..."
    log_info "Version: $ROCKSDB_VERSION"
    log_info "Install prefix: $INSTALL_PREFIX"
    log_info "Build directory: $BUILD_DIR"
    log_info "Number of cores: $NUM_CORES"
    
    # Show sanitizer configuration
    if [ "$USE_ASAN" = true ] || [ "$USE_MSAN" = true ] || [ "$USE_TSAN" = true ] || [ "$USE_UBSAN" = true ]; then
        log_info "Sanitizer configuration:"
        [ "$USE_ASAN" = true ] && log_info "  - AddressSanitizer (ASAN): ENABLED"
        [ "$USE_MSAN" = true ] && log_info "  - MemorySanitizer (MSAN): ENABLED"
        [ "$USE_TSAN" = true ] && log_info "  - ThreadSanitizer (TSAN): ENABLED"
        [ "$USE_UBSAN" = true ] && log_info "  - UndefinedBehaviorSanitizer (UBSAN): ENABLED"
    else
        log_info "Sanitizer configuration: NONE (standard build)"
    fi
    echo
    
    # Set up trap for cleanup
    trap cleanup EXIT
    
    # Run installation steps
    check_root
    check_dependencies
    download_rocksdb
    build_rocksdb
    
    if verify_installation; then
        log_success "RocksDB installation completed successfully!"
        log_info "Libraries installed to: $INSTALL_PREFIX/lib"
        log_info "Headers installed to: $INSTALL_PREFIX/include"
        if [ "$USE_ASAN" = true ] || [ "$USE_MSAN" = true ] || [ "$USE_TSAN" = true ] || [ "$USE_UBSAN" = true ]; then
            log_info "Build includes sanitizers for debugging memory issues"
        fi
        log_info "You can now build your project with RocksDB support"
    else
        log_error "RocksDB installation verification failed"
        exit 1
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                echo "RocksDB Installation Script"
                echo
                echo "Usage: $0 [options]"
                echo
                echo "Options:"
                echo "  -h, --help           Show this help message"
                echo "  -v, --version        Show RocksDB version to install"
                echo "  --asan               Build with AddressSanitizer"
                echo "  --msan               Build with MemorySanitizer"
                echo "  --tsan               Build with ThreadSanitizer"
                echo "  --ubsan              Build with UndefinedBehaviorSanitizer"
                echo "  --all-sanitizers     Build with all sanitizers (ASAN + UBSAN)"
                echo
                echo "Sanitizer Options:"
                echo "  --asan               Detect memory errors (buffer overflows, use-after-free)"
                echo "  --msan               Detect uninitialized memory reads"
                echo "  --tsan               Detect data races in multi-threaded programs"
                echo "  --ubsan              Detect undefined behavior (integer overflow, etc.)"
                echo "  --all-sanitizers     Enable ASAN + UBSAN (most common combination)"
                echo
                echo "This script will:"
                echo "  1. Check and install system dependencies"
                echo "  2. Download RocksDB source code"
                echo "  3. Build RocksDB with C++20 support"
                echo "  4. Install to $INSTALL_PREFIX"
                echo "  5. Verify the installation"
                echo
                echo "Required permissions: sudo (for installing dependencies and RocksDB)"
                echo
                echo "Examples:"
                echo "  $0                           # Standard installation"
                echo "  $0 --asan                    # With AddressSanitizer"
                echo "  $0 --all-sanitizers          # With ASAN + UBSAN"
                echo "  $0 --asan --ubsan            # With ASAN + UBSAN"
                exit 0
                ;;
            -v|--version)
                echo "RocksDB Version: $ROCKSDB_VERSION"
                exit 0
                ;;
            --asan)
                USE_ASAN=true
                shift
                ;;
            --msan)
                USE_MSAN=true
                shift
                ;;
            --tsan)
                USE_TSAN=true
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
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Run main function
main "$@"
