
#include "VecSim/memory/redismodule.h"
#include "VecSim/vec_sim.h"
#include <stdbool.h>
#include <errno.h>
#include <string.h>

#define DIMENSION 100

long long _get_memory_usage (RedisModuleCtx *ctx) {
    long long usage;
    RedisModuleServerInfoData *info = RedisModule_GetServerInfo(ctx,"memory");
    RedisModuleString *memory = RedisModule_ServerInfoGetField(ctx,info,"used_memory");
    RedisModule_StringToLongLong(memory,&usage);
    RedisModule_FreeServerInfo(ctx,info);
    RedisModule_FreeString(ctx,memory);
    return usage;
}

void _fill_vectors(VecSimIndex *index, long long amount) {
    VecSimIndexInfo indexInfo = VecSimIndex_Info(index);
    int i;
    int64_t vec[indexInfo.d];
    for (i=0;i<indexInfo.d;i++)
        vec[i] = i;
    for (i=0;i<amount;i++)
        VecSimIndex_AddVector(index,vec,i);
}

void _delete_vectors(VecSimIndex *index, long long amount) {
    int i;
    for (i=0;i<amount;i++)
        VecSimIndex_DeleteVector(index,i);
}

VecSimIndex* _create_index (VecSimAlgo algo) {

    VecSimParams param;
    param.algo = algo;
    param.bfParams.blockSize = 1;
    param.bfParams.initialCapacity = 1;
    param.type = VecSimType_INT64;
    param.size = DIMENSION;
    param.metric = VecSimMetric_L2;
    
    return VecSimIndex_New(&param);
}

int VecSim_memory_create_index_add_n_delete_m_check (RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

    long long n,m,i;
    long long startMemory,endMemory;
    VecSimAlgo algo = -1;
    RedisModuleString *al;

    if (argc != 4 || RedisModule_StringToLongLong(argv[2],&n) == REDISMODULE_ERR
                  || RedisModule_StringToLongLong(argv[3],&m) == REDISMODULE_ERR) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }
    al = RedisModule_CreateString(ctx,"BF",2);
    if (!RedisModule_StringCompare(argv[1],al))
        algo = VecSimAlgo_BF;
    RedisModule_FreeString(ctx,al);
    al = RedisModule_CreateString(ctx,"HNSW",4);
    if (!RedisModule_StringCompare(argv[1],al))
        algo = VecSimAlgo_HNSWLIB;
    RedisModule_FreeString(ctx,al);
    if (algo == -1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    startMemory = _get_memory_usage(ctx);

    VecSimIndex *index = _create_index(algo);
    _fill_vectors(index,n);
    _delete_vectors(index,m);
    
    endMemory = _get_memory_usage(ctx);

    VecSimIndexInfo indexInfo = VecSimIndex_Info(index);

    // RedisModule_ReplyWithArray(ctx, 3);
    // RedisModule_ReplyWithLongLong(ctx,endMemory-startMemory);
    // RedisModule_ReplyWithLongLong(ctx,indexInfo.memory);

    if (indexInfo.memory <= endMemory-startMemory)
        RedisModule_ReplyWithSimpleString(ctx, "OK");
    else
        RedisModule_ReplyWithError(ctx, "ERROR");

    VecSimIndex_Free(index);
    return REDISMODULE_OK;
}

int VecSim_memory_create_index_add_n_check (RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    int i, res;
    RedisModuleString *newArgv[argc+1];
    for (i=0;i<argc;i++)
        newArgv[i] = argv[i];
    newArgv[argc] = RedisModule_CreateString(ctx,"0",1);
    res = VecSim_memory_create_index_add_n_delete_m_check(ctx,newArgv,argc+1);
    RedisModule_FreeString(ctx,newArgv[argc]);
    return res;
}

int VecSim_memory_create_index_check (RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    int i, res;
    RedisModuleString *newArgv[argc+1];
    for (i=0;i<argc;i++)
        newArgv[i] = argv[i];
    newArgv[argc] = RedisModule_CreateString(ctx,"0",1);
    res = VecSim_memory_create_index_add_n_check(ctx,newArgv,argc+1);
    RedisModule_FreeString(ctx,newArgv[argc]);
    return res;
}

int VecSim_memory_basic_check (RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    VecSimIndex *index = _create_index(VecSimAlgo_BF);

    if (index) {
        RedisModule_ReplyWithSimpleString(ctx, "OK");
        VecSimIndex_Free(index);
    }
    else
        RedisModule_ReplyWithError(ctx, "ERROR");

    return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if (RedisModule_Init(ctx, "VecSim_memory", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "VecSim_memory.basic_check", VecSim_memory_basic_check,
                                  "", 0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "VecSim_memory.create_index_check", VecSim_memory_create_index_check,
                                  "", 0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;
    
    if (RedisModule_CreateCommand(ctx, "VecSim_memory.create_index_add_n_check", VecSim_memory_create_index_add_n_check,
                                  "", 0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;
    
    if (RedisModule_CreateCommand(ctx, "VecSim_memory.create_index_add_n_delete_m_check", VecSim_memory_create_index_add_n_delete_m_check,
                                  "", 0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    return REDISMODULE_OK;
}