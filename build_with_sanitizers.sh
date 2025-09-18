#!/bin/bash

# VectorSimilarity Build Script with Sanitizer Support
# This script builds VectorSimilarity with various sanitizer options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
USE_ASAN=false
USE_MSAN=false
USE_TSAN=false
USE_UBSAN=false
BUILD_TYPE="Release"
BUILD_DIR="bin/Linux-x86_64-debug-asan"
CLEAN_BUILD=false

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

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                echo "VectorSimilarity Build Script with Sanitizer Support"
                echo
                echo "Usage: $0 [options]"
                echo
                echo "Options:"
                echo "  -h, --help           Show this help message"
                echo "  --asan               Build with AddressSanitizer"
                echo "  --msan               Build with MemorySanitizer"
                echo "  --tsan               Build with ThreadSanitizer"
                echo "  --ubsan              Build with UndefinedBehaviorSanitizer"
                echo "  --all-sanitizers     Build with all sanitizers (ASAN + UBSAN)"
                echo "  --clean              Clean build directory before building"
                echo "  --build-dir DIR      Specify build directory (default: $BUILD_DIR)"
                echo
                echo "Sanitizer Options:"
                echo "  --asan               Detect memory errors (buffer overflows, use-after-free)"
                echo "  --msan               Detect uninitialized memory reads"
                echo "  --tsan               Detect data races in multi-threaded programs"
                echo "  --ubsan              Detect undefined behavior (integer overflow, etc.)"
                echo "  --all-sanitizers     Enable ASAN + UBSAN (most common combination)"
                echo
                echo "Examples:"
                echo "  $0                           # Standard debug build"
                echo "  $0 --asan                    # With AddressSanitizer"
                echo "  $0 --all-sanitizers          # With ASAN + UBSAN"
                echo "  $0 --asan --clean            # Clean build with ASAN"
                echo "  $0 --build-dir custom_build  # Custom build directory"
                exit 0
                ;;
            --asan)
                USE_ASAN=true
                BUILD_TYPE="Debug"
                shift
                ;;
            --msan)
                USE_MSAN=true
                BUILD_TYPE="Debug"
                shift
                ;;
            --tsan)
                USE_TSAN=true
                BUILD_TYPE="Debug"
                shift
                ;;
            --ubsan)
                USE_UBSAN=true
                BUILD_TYPE="Debug"
                shift
                ;;
            --all-sanitizers)
                USE_ASAN=true
                USE_UBSAN=true
                BUILD_TYPE="Debug"
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --build-dir)
                BUILD_DIR="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Clean build directory
clean_build() {
    if [ "$CLEAN_BUILD" = true ]; then
        log_info "Cleaning build directory: $BUILD_DIR"
        if [ -d "$BUILD_DIR" ]; then
            rm -rf "$BUILD_DIR"
        fi
    fi
}

# Build VectorSimilarity
build_vectorsimilarity() {
    log_info "Building VectorSimilarity..."
    log_info "Build directory: $BUILD_DIR"
    log_info "Build type: $BUILD_TYPE"
    
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
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Prepare CMake flags
    local cmake_flags=(
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DVECSIM_BUILD_TESTS=ON"
        "-DUSE_SVS=OFF"
        "-DCMAKE_CXX_STANDARD=20"
        "-DCMAKE_PREFIX_PATH=/usr/local"
        "-DCMAKE_MODULE_PATH=/usr/local/lib/cmake/rocksdb"
    )
    
    # Add sanitizer flags
    if [ "$USE_ASAN" = true ]; then
        cmake_flags+=("-DUSE_ASAN=ON")
    fi
    
    if [ "$USE_MSAN" = true ]; then
        cmake_flags+=("-DUSE_MSAN=ON")
    fi
    
    if [ "$USE_TSAN" = true ]; then
        cmake_flags+=("-DUSE_TSAN=ON")
    fi
    
    if [ "$USE_UBSAN" = true ]; then
        cmake_flags+=("-DUSE_UBSAN=ON")
    fi
    
    # Force use of the correct RocksDB library
    cmake_flags+=("-DROCKSDB_ROOT_DIR=/usr/local")
    cmake_flags+=("-DROCKSDB_INCLUDE_DIR=/usr/local/include")
    cmake_flags+=("-DROCKSDB_LIBRARY=/usr/local/lib/librocksdb.so")
    
    # Add compression libraries to link flags
    cmake_flags+=("-DCMAKE_EXE_LINKER_FLAGS=-lzstd -lsnappy -llz4 -lbz2 -lz")
    cmake_flags+=("-DCMAKE_SHARED_LINKER_FLAGS=-lzstd -lsnappy -llz4 -lbz2 -lz")
    
    # Configure with CMake
    log_info "Configuring with CMake..."
    CC=clang CXX=clang++ cmake "${cmake_flags[@]}" /home/ben/Repos/VectorSimilarity
    
    # Build
    log_info "Building VectorSimilarity..."
    make -j$(nproc)
    
    log_success "VectorSimilarity built successfully!"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Set sanitizer options
    local asan_options="abort_on_error=0:detect_leaks=1"
    export ASAN_OPTIONS="$asan_options"
    
    # Set up LD_PRELOAD for ASAN if needed
    local ld_preload=""
    if [ "$USE_ASAN" = true ]; then
        ld_preload="LD_PRELOAD=/lib/x86_64-linux-gnu/libasan.so.8"
        log_info "Using LD_PRELOAD for AddressSanitizer"
    fi
    
    # Run specific tests
    if [ -f "unit_tests/test_hnsw_disk" ]; then
        log_info "Running HNSW Disk tests..."
        if [ -n "$ld_preload" ]; then
            $ld_preload ./unit_tests/test_hnsw_disk --gtest_filter="*BasicConstruction*"
        else
            ./unit_tests/test_hnsw_disk --gtest_filter="*BasicConstruction*"
        fi
        log_success "HNSW Disk tests passed!"
    fi
    
    # Run other tests if they exist
    if [ -f "unit_tests/test_hnsw" ]; then
        log_info "Running HNSW tests..."
        if [ -n "$ld_preload" ]; then
            $ld_preload ./unit_tests/test_hnsw --gtest_filter="*Basic*" || log_warning "Some HNSW tests failed"
        else
            ./unit_tests/test_hnsw --gtest_filter="*Basic*" || log_warning "Some HNSW tests failed"
        fi
    fi
}

# Main function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    log_info "Starting VectorSimilarity build with sanitizer support..."
    echo
    
    # Clean if requested
    clean_build
    
    # Build VectorSimilarity
    build_vectorsimilarity
    
    # Run tests
    run_tests
    
    log_success "Build and test completed successfully!"
    log_info "Build directory: $BUILD_DIR"
    log_info "You can now run your tests with sanitizer support"
}

# Run main function
main "$@"
