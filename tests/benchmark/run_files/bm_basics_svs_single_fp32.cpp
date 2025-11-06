#include "benchmark/bm_vecsim_basics.h"
#include "benchmark/bm_svs_index.h"
#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/hnsw/hnsw_single.h"

/**************************************
  Basic tests for single value index with fp32 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = false;
uint32_t BM_VecSimGeneral::enabled_index_types = IndexTypeFlags::INDEX_MASK_SVS
    // #if HAVE_SVS_LVQ
    //     | IndexTypeFlags::INDEX_MASK_SVS_LVQ8
    // #endif
    ;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 128;
size_t BM_VecSimGeneral::EF_C = 512;

const char *BM_VecSimGeneral::hnsw_index_file = nullptr;
template <>
const char *BM_VecSimSVSdIndex<fp32_index_t>::svs_index_tar_file =
    "tests/benchmark/data/svs-dbpedia-cosine-dim768-quant-none.tar.gz";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(bm_func, algo, Single)
#define BM_ADD_LABEL                CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, Single)

#include "benchmark/bm_initialization/bm_basics_svs_initialize_fp32.h"
BENCHMARK_MAIN();
