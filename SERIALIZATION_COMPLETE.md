# DBpedia Dataset Serialization - Complete ✅

## Summary

Successfully serialized the DBpedia dataset for use in HNSW disk benchmarks!

## What Was Done

### 1. Fixed Serializer Output Format

Updated `HNSWDiskIndex::saveIndex` to create the correct directory structure:

```
output_folder/
├── index.hnsw_disk_v1    (metadata file)
└── rocksdb/               (checkpoint directory)
```

**Key changes:**
- Modified `HNSWDiskIndex::getCheckpointDir()` to handle both file and folder paths
- Modified `HNSWDiskIndex::saveIndex()` to detect if location is a folder or file
- If folder: creates `folder/index.hnsw_disk_v1` and `folder/rocksdb/`
- If file: creates file and `parent_dir/rocksdb/`

### 2. Serialized DBpedia Dataset

**Input:**
- File: `tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw`
- Vectors: 10,000
- Dimension: 768
- Type: float32
- Metric: Cosine

**Command:**
```bash
cd build/hnsw_disk_serializer
./hnsw_disk_serializer \
  ../../tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw \
  ../../tests/benchmark/data/dbpedia-cosine-dim768-M64-efc512-disk \
  768 Cosine FLOAT32 64 512
```

**Output:**
```
tests/benchmark/data/dbpedia-cosine-dim768-M64-efc512-disk/
├── index.hnsw_disk_v1 (284 KB)
└── rocksdb/ (31 MB)
```

**Performance:**
- Indexing time: ~22 seconds
- Total size: ~31 MB

### 3. Updated Benchmark Configuration

Modified `tests/benchmark/run_files/bm_hnsw_disk_single_fp32.cpp`:

```cpp
// Configure using dbpedia dataset (10K vectors, 768 dimensions)
size_t BM_VecSimGeneral::n_queries = 100;
size_t BM_VecSimGeneral::n_vectors = 10000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;

// Set the HNSW disk index folder path (loads pre-serialized index)
template <>
std::string BM_VecSimIndex<fp32_index_t>::hnsw_disk_index_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-M64-efc512-disk";
```

## How to Use in Benchmarks

The benchmark will now:
1. Load the pre-serialized index from the folder
2. Open RocksDB from `folder/rocksdb/`
3. Read metadata from `folder/index.hnsw_disk_v1`
4. Run queries against the disk-based index

**To run the benchmark:**
```bash
cd build
make benchmark
./benchmark/bm_hnsw_disk_single_fp32
```

## Directory Structure

```
tests/benchmark/data/
├── dbpedia-cosine-dim768-test_vectors.raw          (raw vectors - 30 MB)
├── dbpedia-cosine-dim768-M64-efc512.hnsw_v3        (in-memory HNSW - 3.1 GB)
└── dbpedia-cosine-dim768-M64-efc512-disk/          (disk-based HNSW - 31 MB)
    ├── index.hnsw_disk_v1                          (metadata - 284 KB)
    └── rocksdb/                                     (checkpoint - 31 MB)
        ├── 000004.log
        ├── 000008.sst
        ├── CURRENT
        ├── MANIFEST-000005
        └── OPTIONS-000007
```

## Next Steps

1. **Run the benchmark** to test the disk-based HNSW index
2. **Compare performance** with in-memory HNSW
3. **Serialize other datasets** using the same approach

## Serializing Other Datasets

To serialize other datasets, use the same pattern:

```bash
./hnsw_disk_serializer \
  <input.raw> \
  <output_folder> \
  <dim> \
  <metric> \
  <type> \
  [M] \
  [efConstruction]
```

**Example for SIFT dataset:**
```bash
./hnsw_disk_serializer \
  sift-128-euclidean.raw \
  sift-128-euclidean-M32-efc200-disk \
  128 L2 FLOAT32 32 200
```

This will create:
```
sift-128-euclidean-M32-efc200-disk/
├── index.hnsw_disk_v1
└── rocksdb/
```

