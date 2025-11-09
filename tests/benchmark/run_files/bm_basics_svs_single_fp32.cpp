#include "benchmark/bm_vecsim_svs.h"

/**************************************
  Basic tests for single value index with fp32 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = false;

size_t BM_VecSimGeneral::n_queries = 20000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 128;
size_t BM_VecSimGeneral::EF_C = 512;
size_t BM_VecSimGeneral::block_size = 1024;

#define DATA_TYPE_INDEX_T fp32_index_t
#define QUANT_BITS_ARG    VecSimSvsQuant_NONE
template <>
const char *BM_VecSimSVS<DATA_TYPE_INDEX_T>::svs_index_tar_file =
    "tests/benchmark/data/svs-dbpedia-cosine-dim768-quant-none.tar.gz";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-1M-vectors.raw";

#include "benchmark/bm_initialization/bm_basics_svs_initialize_fp32.h"
BENCHMARK_MAIN();
