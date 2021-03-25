#define REDISMODULE_MAIN
#include "redismodule.h"
#include <stdlib.h>
#include <time.h>
#include "../../src/hnsw_c.h"

int hnswlib_vector_add_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
    HNSWIndex *index = InitHNSWIndex(200, 4);
    if (GetHNSWIndexSize(index) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "Init error");
    }
    float a[4] = {1.0, 1.0, 1.0, 1.0};
    AddVectorToHNSWIndex(index, (const void *)a, 1);
    if (GetHNSWIndexSize(index) != 1) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int hnswlib_vector_search_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
    HNSWIndex *index = InitHNSWIndex(200, 4);

    for (float i = 0; i < 100; i++) {
        float f[4] = {i, i, i, i};
        AddVectorToHNSWIndex(index, (const void *)f, i);
    }
    if (GetHNSWIndexSize(index) != 100) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    float query[4] = {50, 50, 50, 50};
    Vector *res = HNSWSearch(index,  (const void *)query, 11);
    for (int i=0; i<11; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].dist;
        if ((diff_id != (i+1)/2) || (dist != (4*((i+1)/2)*((i+1)/2)))) {
            return RedisModule_ReplyWithSimpleString(ctx, "Search test fail");
        }
    }
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if(RedisModule_Init(ctx, "vec_sim_test", 1, REDISMODULE_APIVER_1) ==
       REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if(
      RedisModule_CreateCommand(ctx, "vec_sim_test.hnswlib_vector_add", hnswlib_vector_add_test,
        "",
        0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if(
      RedisModule_CreateCommand(ctx, "vec_sim_test.hnswlib_search", hnswlib_vector_search_test,
        "",
        0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;
}
