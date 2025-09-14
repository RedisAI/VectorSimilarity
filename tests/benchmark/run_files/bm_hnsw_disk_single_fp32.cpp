#include "benchmark/bm_vecsim_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_disk.h"

/**************************************
  Basic tests for disk-based HNSW index with fp32 data type.
  This benchmark will test the new HNSWDiskIndex implementation.
***************************************/

bool BM_VecSimGeneral::is_multi = false;
uint32_t BM_VecSimGeneral::enabled_index_types = DEFAULT_BM_INDEXES_MASK;

// Configure using existing dbpedia dataset for now (can be changed to SIFT1B later)
size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 768;          // dbpedia vectors are 768-dimensional
size_t BM_VecSimGeneral::M = 64;             // HNSW parameter M
size_t BM_VecSimGeneral::EF_C = 512;         // HNSW construction parameter

// Dataset file paths - using existing dbpedia dataset
const char *BM_VecSimGeneral::hnsw_index_file = "tests/benchmark/data/dbpedia-cosine-dim768-M64-efc512.hnsw_v3";
const char *BM_VecSimGeneral::test_queries_file = "tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(bm_func, algo, Single)
#define BM_ADD_LABEL                CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, Single)
#define BM_ADD_LABEL_ASYNC          CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, Async, Single)
#define BM_DELETE_LABEL_ASYNC       CONCAT_WITH_UNDERSCORE_ARCH(DeleteLabel_Async, Single)

// Define benchmarks for different index types
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, BF), fp32_index_t, BruteForceIndex_Single, float,
                    float, INDEX_BF)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, HNSW), fp32_index_t, HNSWIndex_Single, float, float,
                    INDEX_HNSW)
DEFINE_DELETE_LABEL(BM_FUNC_NAME(DeleteLabel, HNSWDisk), fp32_index_t, HNSWDiskIndex, float, float,
                    INDEX_HNSW_DISK)

#include "benchmark/bm_initialization/bm_hnsw_disk_initialize_fp32.h"
BENCHMARK_MAIN();
