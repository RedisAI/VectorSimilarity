#define REDISMODULE_MAIN
#include "redismodule.h"
#include "../../src/hnsw_c.h"

int nmslib_vector_add_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
    HNSWIndex *index = InitHNSWIndex();
    if (GetHNSWIndexSize(index) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "Init error");
    }
    float a = 1.0f;
    char a_as_bytes[4];
    *((float *)a_as_bytes) = a;
    AddVectorToHNSWIndex(index, a_as_bytes, 1);
    if (GetHNSWIndexSize(index) != 1) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if(RedisModule_Init(ctx, "vector_similarity_test", 1, REDISMODULE_APIVER_1) ==
       REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if(
      RedisModule_CreateCommand(ctx, "nmslib_vector_add_test", nmslib_vector_add_test,
        "",
        0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;
}
