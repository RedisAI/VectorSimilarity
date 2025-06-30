#include "benchmark/bm_vecsim_basics.h"

#include "VecSim/algorithms/brute_force/brute_force_single.h"
#include "VecSim/algorithms/svs/svs_tiered.h"
#include "VecSim/algorithms/svs/svs.h"

// Type alias to make SVS compatible with benchmark framework
// Uses default values for QuantBits=0, ResidualBits=0, IsLeanVec=false (no compression)
// Uses DistanceIP for Cosine metric (as per SVS factory mapping)
template <typename DataType, typename DistType>
using SVSIndex_Single_BM = SVSIndex_Single<DistType, DataType, 0, 0, false>;

// Type alias to make TieredSVSIndex compatible with benchmark framework
// TieredSVSIndex only takes DataType parameter, DistType is ignored (always float)
template <typename DataType, typename DistType>
using TieredSVSIndex_Single = TieredSVSIndex<DataType>;

/**************************************
  Basic tests for single value index with fp32 data type.
***************************************/

bool BM_VecSimGeneral::is_multi = false;
uint32_t BM_VecSimGeneral::enabled_index_types =
    IndexTypeFlags::INDEX_TYPE_BF |
    IndexTypeFlags::INDEX_TYPE_HNSW | // For now the HNSW index is required
    IndexTypeFlags::INDEX_TYPE_SVS | IndexTypeFlags::INDEX_TYPE_TIERED_SVS;

size_t BM_VecSimGeneral::n_queries = 10000;
size_t BM_VecSimGeneral::n_vectors = 1000000;
size_t BM_VecSimGeneral::dim = 768;
size_t BM_VecSimGeneral::M = 64;
size_t BM_VecSimGeneral::EF_C = 512;
tieredIndexMock BM_VecSimGeneral::mock_thread_pool{};

const char *BM_VecSimGeneral::hnsw_index_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-M64-efc512.hnsw_v3";
const char *BM_VecSimGeneral::svs_index_folder =
    "tests/benchmark/data/dbpedia_svs/";
const char *BM_VecSimGeneral::test_queries_file =
    "tests/benchmark/data/dbpedia-cosine-dim768-test_vectors.raw";

#define BM_FUNC_NAME(bm_func, algo) CONCAT_WITH_UNDERSCORE_ARCH(bm_func, algo, Single)
#define BM_ADD_LABEL                CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, SVS, Single)
#define BM_ADD_LABEL_ASYNC          CONCAT_WITH_UNDERSCORE_ARCH(AddLabel, Async, SVS, Single)
#define BM_DELETE_LABEL_ASYNC       CONCAT_WITH_UNDERSCORE_ARCH(DeleteLabel_Async, SVS, Single)

DEFINE_DELETE_LABEL_SVS(BM_FUNC_NAME(DeleteLabel, SVS), fp32_index_t, SVSIndex_Single_BM, float, float,
                        INDEX_SVS)
DEFINE_DELETE_LABEL_SVS(BM_FUNC_NAME(DeleteLabel, Tiered), fp32_index_t, TieredSVSIndex_Single,
                        float, float, INDEX_TIERED_SVS)

#include "benchmark/bm_initialization/SVS_bm_basics_initialize_fp32.h"
BENCHMARK_MAIN();
