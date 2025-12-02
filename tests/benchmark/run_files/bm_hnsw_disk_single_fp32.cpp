#include "benchmark/bm_vecsim_basics.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"
#include "VecSim/algorithms/hnsw/hnsw_disk.h"

/**************************************
  Basic tests for disk-based HNSW index with fp32 data type.
  This benchmark will test the new HNSWDiskIndex implementation.
***************************************/

bool BM_VecSimGeneral::is_multi = false;
// Only enable HNSW_DISK for this benchmark
uint32_t BM_VecSimGeneral::enabled_index_types = IndexTypeFlags::INDEX_MASK_HNSW_DISK;

// Configure using deep dataset (1M vectors, 96 dimensions)
size_t BM_VecSimGeneral::n_queries = 100;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 96;
size_t BM_VecSimGeneral::M = 32;
size_t BM_VecSimGeneral::EF_C = 256;

// Dataset file paths - using deep dataset
// For HNSW disk, hnsw_index_file points to the folder containing index.hnsw_disk_v1 and rocksdb/
const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/deep-10M-cosine-dim96-M32-efc200-disk-vectors";
const char *BM_VecSimGeneral::test_queries_file = "tests/benchmark/data/deep.query.public.10K.fbin";
const char *BM_VecSimGeneral::ground_truth_file = "tests/benchmark/data/deep.groundtruth.10M.10K.ibin"; // defined only for this benchmark

#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(bm_func, algo, Single)
#define BM_ADD_LABEL                CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, Single)
#define BM_ADD_LABEL_ASYNC          CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, Async, Single)
#define BM_DELETE_LABEL_ASYNC       CONCAT_WITH_UNDERSCORE_ARCH(DeleteLabel_Async, Single)
#define BM_FLUSH_BATCH_DISK         CONCAT_WITH_UNDERSCORE_ARCH(FlushBatchDisk, Single)

#include "benchmark/bm_initialization/bm_hnsw_disk_initialize_fp32.h"
BENCHMARK_MAIN();
