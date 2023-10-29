
/*
 *Copyright Redis Ltd. 2021 - present
 *Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 *the Server Side Public License v1 (SSPLv1).
 */

#include "redismodule.h"
#include "VecSim/vec_sim.h"
#include <stdbool.h>
#include <errno.h>
#include <string.h>

#define DIMENSION 100

// Get memory usage from redis-server.
long long _get_memory_usage(RedisModuleCtx *ctx) {
    long long usage;
    RedisModuleServerInfoData *info = RedisModule_GetServerInfo(ctx, "memory");
    RedisModuleString *memory = RedisModule_ServerInfoGetField(ctx, info, "used_memory");
    RedisModule_StringToLongLong(memory, &usage);
    RedisModule_FreeServerInfo(ctx, info);
    RedisModule_FreeString(ctx, memory);
    return usage;
}

// Adds 'amount' vectors to the index. could be 0.
void _add_vectors(VecSimIndex *index, long long amount) {
    VecSimIndexInfo indexInfo = VecSimIndex_Info(index);
    size_t dim = indexInfo.commonInfo.basicInfo.dim;
    double vec[dim];
    for (int i = 0; i < dim; i++)
        vec[i] = i;
    for (long long j = 0; j < amount; j++)
        VecSimIndex_AddVector(index, vec, j);
}

// Deletes 'amount' vectors from the index. could be 0 or larger from the number of vectors in the
// index.
void _delete_vectors(VecSimIndex *index, long long amount) {
    for (long long i = 0; i < amount; i++)
        VecSimIndex_DeleteVector(index, i);
}

// Creates a generic index, supports Brute Force and HNSW.
VecSimIndex *_create_index(VecSimAlgo algo) {

    VecSimParams param = {0};
    param.algo = algo;
    switch (algo) {
    case VecSimAlgo_BF:
        param.algoParams.bfParams.blockSize = 1;
        param.algoParams.bfParams.initialCapacity = 1;
        param.algoParams.bfParams.type = VecSimType_FLOAT64;
        param.algoParams.bfParams.dim = DIMENSION;
        param.algoParams.bfParams.metric = VecSimMetric_L2;
        param.algoParams.bfParams.multi = false;
        break;

    case VecSimAlgo_HNSWLIB:
        param.algoParams.hnswParams.M = 30;
        param.algoParams.hnswParams.initialCapacity = 1;
        param.algoParams.hnswParams.efConstruction = 0;
        param.algoParams.hnswParams.efRuntime = 0;
        param.algoParams.hnswParams.type = VecSimType_FLOAT64;
        param.algoParams.hnswParams.dim = DIMENSION;
        param.algoParams.hnswParams.metric = VecSimMetric_L2;
        param.algoParams.hnswParams.multi = false;
        break;
    // TODO: add memory test for tiered index
    case VecSimAlgo_TIERED:
        return NULL;
    }

    return VecSimIndex_New(&param);
}

// Returns the VecSimAlgo enum value corresponding to the name specified in 'algo'.
// Please keep up-to-date if additional algorithms are implemented.
VecSimAlgo _get_algorithm(RedisModuleString *algo) {

    const char *al = RedisModule_StringPtrLen(algo, NULL);
    if (!strcmp(al, "BF"))
        return VecSimAlgo_BF;
    if (!strcmp(al, "HNSW"))
        return VecSimAlgo_HNSWLIB;

    return -1;
}

// This is the test function used by all other tests.
// Creates an index, add 'addNum' vectors, deletes 'delNum' vectors and then compares redis-server
// memory usage report to the index memory usage report.
// Uncomment the 3 lines close to the end to get the values along with the comparison result.
int _VecSim_memory_create_check_impl(RedisModuleCtx *ctx, VecSimAlgo algo, long long addNum,
                                     long long delNum) {

    long long startMemory, endMemory;
    startMemory = _get_memory_usage(ctx); // Gets memory usage before creating the index.

    VecSimIndex *index = _create_index(algo);
    _add_vectors(index, addNum);
    _delete_vectors(index, delNum);

    endMemory = _get_memory_usage(ctx); // Gets memory usage after creating the index.

    VecSimIndexInfo indexInfo = VecSimIndex_Info(index);

    // RedisModule_ReplyWithArray(ctx, 3);
    // RedisModule_ReplyWithLongLong(ctx,endMemory-startMemory);
    // RedisModule_ReplyWithLongLong(ctx,indexInfo.memory);

    // Actual test: verify that memory usage known to the server is at least the memory amount used
    // by the index.
    uint64_t memory = indexInfo.commonInfo.memory;

    if (memory <= endMemory - startMemory)
        RedisModule_ReplyWithSimpleString(ctx, "OK");
    else
        RedisModule_ReplyWithError(ctx, "ERROR: some undeclared memory allocation detected.");

    VecSimIndex_Free(index);
    return REDISMODULE_OK;
}

// Calls to _VecSim_memory_create_check_impl with algo,n,m.
int VecSim_memory_create_index_add_n_delete_m_check(RedisModuleCtx *ctx, RedisModuleString **argv,
                                                    int argc) {
    if (argc != 4) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    VecSimAlgo algo;
    if ((algo = _get_algorithm(argv[1])) == -1) {
        RedisModule_ReplyWithError(ctx, "ERROR: First argument should be a supported algorithm.");
        return REDISMODULE_OK;
    }

    long long n;
    if (RedisModule_StringToLongLong(argv[2], &n) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx,
                                   "ERROR: Second argument should be a number of vectors to add.");
        return REDISMODULE_OK;
    }

    long long m;
    if (RedisModule_StringToLongLong(argv[3], &m) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(
            ctx, "ERROR: Third argument should be a number of vectors to delete.");
        return REDISMODULE_OK;
    }

    return _VecSim_memory_create_check_impl(ctx, algo, n, m);
}

// Calls to _VecSim_memory_create_check_impl with algo,n,0.
int VecSim_memory_create_index_add_n_check(RedisModuleCtx *ctx, RedisModuleString **argv,
                                           int argc) {
    if (argc != 3) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    VecSimAlgo algo;
    if ((algo = _get_algorithm(argv[1])) == -1) {
        RedisModule_ReplyWithError(ctx, "ERROR: First argument should be a supported algorithm.");
        return REDISMODULE_OK;
    }

    long long n;
    if (RedisModule_StringToLongLong(argv[2], &n) == REDISMODULE_ERR) {
        RedisModule_ReplyWithError(ctx,
                                   "ERROR: Second argument should be a number of vectors to add.");
        return REDISMODULE_OK;
    }

    return _VecSim_memory_create_check_impl(ctx, algo, n, 0);
}

// Calls to _VecSim_memory_create_check_impl with algo,0,0.
int VecSim_memory_create_index_check(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

    if (argc != 2) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    VecSimAlgo algo;
    if ((algo = _get_algorithm(argv[1])) == -1) {
        RedisModule_ReplyWithError(ctx, "ERROR: First argument should be a supported algorithm.");
        return REDISMODULE_OK;
    }

    return _VecSim_memory_create_check_impl(ctx, algo, 0, 0);
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if (RedisModule_Init(ctx, "VecSim_memory", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    // Usage: VecSim_memory.create_index_check <algorithm:BF/HNSW>.
    if (RedisModule_CreateCommand(ctx, "VecSim_memory.create_index_check",
                                  VecSim_memory_create_index_check, "", 0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    // Usage: VecSim_memory.create_index_add_n_check <algorithm:BF/HNSW> <number>.
    if (RedisModule_CreateCommand(ctx, "VecSim_memory.create_index_add_n_check",
                                  VecSim_memory_create_index_add_n_check, "", 0, 0,
                                  0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    // Usage: VecSim_memory.create_index_add_n_delete_m_check <algorithm:BF/HNSW> <number> <number>.
    if (RedisModule_CreateCommand(ctx, "VecSim_memory.create_index_add_n_delete_m_check",
                                  VecSim_memory_create_index_add_n_delete_m_check, "", 0, 0,
                                  0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    VecSimMemoryFunctions memoryFunctions = {.allocFunction = RedisModule_Alloc,
                                             .callocFunction = RedisModule_Calloc,
                                             .freeFunction = RedisModule_Free,
                                             .reallocFunction = RedisModule_Realloc};
    VecSim_SetMemoryFunctions(memoryFunctions);

    return REDISMODULE_OK;
}
