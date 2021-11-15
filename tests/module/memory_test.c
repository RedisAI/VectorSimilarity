
#include "redismodule.h"
#include "vec_sim.h"
#include <pthread.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>

int VecSim_memory_basic_check (RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);

    if (argc > 1) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_OK;
    }

    VecSimParams param;
    param.algo = VecSimAlgo_BF;
    param.bfParams.blockSize = 1;
    param.bfParams.initialCapacity = 1;
    param.type = VecSimType_INT32;
    param.size = 1;
    param.metric = VecSimMetric_L2;
    VecSimIndex *vec;
    vec = VecSimIndex_New(&param);

    if (!vec)
        RedisModule_ReplyWithSimpleString(ctx, "OK");
    else
        RedisModule_ReplyWithError(ctx, "ERROR");

    return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {

    if (RedisModule_Init(ctx, "RAI_llapi", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if (RedisModule_CreateCommand(ctx, "VecSim_memory.basic_check", VecSim_memory_basic_check, "", 0, 0,
                                  0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    return REDISMODULE_OK;
}