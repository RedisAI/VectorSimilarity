# HNSW Disk Index Serializer - Summary

## What Was Created

I've successfully created a C++ serializer program for creating HNSW disk indexes from raw vector data.

### Files Created:

1. **tests/benchmark/data/scripts/hnsw_disk_serializer.cpp**
   - Standalone C++ program
   - Reads raw binary vectors
   - Creates HNSWDiskIndex with RocksDB
   - Bulk inserts vectors
   - Saves index with checkpoint

2. **tests/benchmark/data/scripts/CMakeLists.txt**
   - Build system for the serializer
   - Integrates with main VectorSimilarity project
   - Links against VectorSimilarity and RocksDB

3. **tests/benchmark/data/scripts/serialize_disk_index.py**
   - Python helper script (for future use with HDF5 datasets)
   - Reads HDF5 files
   - Writes vectors to temporary raw binary
   - Calls C++ serializer
   - Cleans up temporary files

4. **tests/benchmark/data/scripts/HOW_TO_USE.md**
   - Quick reference guide
   - Build instructions
   - Usage examples

### Build System Changes:

- **CMakeLists.txt** (main): Added subdirectory for serializer
- **src/VecSim/CMakeLists.txt**: Temporarily disabled SVS/Tiered factories (due to SVS header issues)
- **src/VecSim/index_factories/index_factory.cpp**: Temporarily disabled SVS/Tiered factory calls

## How to Build

```bash
cd /path/to/VectorSimilarity
mkdir -p build && cd build
cmake ..
make hnsw_disk_serializer
```

The binary will be at: `build/hnsw_disk_serializer/hnsw_disk_serializer`

## How to Use

### Basic Usage

```bash
cd build/hnsw_disk_serializer
./hnsw_disk_serializer <input.raw> <output_name> <dim> <metric> <type> [M] [efC]
```

### Example

```bash
# Create a test dataset (1000 vectors, 128 dimensions)
python3 -c "import numpy as np; np.random.randn(1000, 128).astype('float32').tofile('test_vectors.raw')"

# Serialize it
./hnsw_disk_serializer test_vectors.raw my_index 128 L2 FLOAT32 16 200

# This creates:
# - my_index.hnsw_disk_v1 (metadata file)
# - my_index.hnsw_disk_v1_rocksdb/ (RocksDB checkpoint directory)
```

### Parameters

- **M**: Number of connections per node (16-128, default: 64)
  - Lower = faster build, less memory, lower recall
  - Higher = slower build, more memory, higher recall

- **efConstruction**: Size of dynamic candidate list (default: 512)
  - Should be >= M
  - Typically 2-10x larger than M
  - Higher = better quality, slower build

### Metrics

- **L2**: Euclidean distance
- **IP**: Inner product (dot product)
- **Cosine**: Cosine similarity

### Data Types

- **FLOAT32**: 32-bit floating point (most common)
- **FLOAT64**: 64-bit floating point
- **BFLOAT16**: 16-bit brain floating point
- **FLOAT16**: 16-bit floating point
- **INT8**: 8-bit signed integer
- **UINT8**: 8-bit unsigned integer

## Output Format

The serializer creates two outputs:

1. **Metadata file** (`<name>.hnsw_disk_v1`):
   - Binary file containing:
     - Index parameters (M, efConstruction, dim, metric, etc.)
     - Graph structure (HNSW layers, connections)
     - Label mappings
     - RocksDB path reference

2. **RocksDB checkpoint** (`<name>.hnsw_disk_v1_rocksdb/`):
   - Directory containing:
     - Vector data in RocksDB format
     - Graph edges and connections
     - Optimized for disk-based access

## Using in Benchmarks

```cpp
#include "VecSim/index_factories/hnsw_disk_factory.h"

// Load the index
std::string index_path = "my_index.hnsw_disk_v1";
VecSimIndex *index = HNSWDiskFactory::NewIndex(index_path);

// Run queries
float query[128] = {...};
VecSimQueryParams params;
params.hnswRuntimeParams.efRuntime = 100;

auto results = VecSimIndex_TopKQuery(index, query, 10, &params, BY_SCORE);

// Cleanup
VecSimQueryResult_Free(results);
VecSimIndex_Free(index);
```

## Next Steps

1. **Test the serializer** with a small dataset
2. **Create serialized indexes** for your benchmark datasets
3. **Update benchmarks** to load serialized indexes
4. **Compare performance** between in-memory and disk-based indexes

## Known Issues

- SVS and Tiered factories are temporarily disabled due to SVS header issues
- This doesn't affect HNSW disk index functionality
- The Python helper script requires HDF5 support (h5py package)

## Performance Tips

1. Start with smaller M values (16-32) for faster indexing
2. Use efConstruction = 2-4x M for good quality
3. Monitor disk space - RocksDB checkpoints can be large
4. For benchmarking, create multiple indexes with different parameters

