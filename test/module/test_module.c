
#define REDISMODULE_MAIN
#include "redismodule.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include "VectorSimilarity/src/vecsim.h"

int hnswlib_vector_add_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
    VecSimAlgoParams params = {
        .hnswParams = {
            .initialCapacity = 200,
            .M = 16,
            .efConstuction = 200
        },
        .algorithmType = VecSimAlgo_HNSW
    };
    VecSimIndex *index = VecSimIndex_New(&params, VecSimMetric_L2, VecSimVecType_FLOAT32, 4);
    if (VecSimIndex_IndexSize(index) != 0) {
        return RedisModule_ReplyWithSimpleString(ctx, "Init error");
    }
    float a[4] = {1.0, 1.0, 1.0, 1.0};
    VecSimIndex_AddVector(index, (const void *)a, 1);
    if (VecSimIndex_IndexSize(index) != 1) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int hnswlib_vector_search_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
    VecSimAlgoParams params = {
        .hnswParams = {
            .initialCapacity = 200,
            .M = 16,
            .efConstuction = 200
        },
        .algorithmType = VecSimAlgo_HNSW
    };
    VecSimIndex *index = VecSimIndex_New(&params, VecSimMetric_L2, VecSimVecType_FLOAT32, 4);

    for (float i = 0; i < 100; i++) {
        float f[4] = {i, i, i, i};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    if (VecSimIndex_IndexSize(index) != 100) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    float query[4] = {50, 50, 50, 50};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index,  (const void *)query, 11);
    for (int i=0; i<11; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].score;
        if ((diff_id != (i+1)/2) || (dist != (4*((i+1)/2)*((i+1)/2)))) {
            return RedisModule_ReplyWithSimpleString(ctx, "Search test fail");
        }
    }
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}


// int hnswlib_index_save_load_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
//     REDISMODULE_NOT_USED(argv);
//     if(argc != 1) {
//         return RedisModule_WrongArity(ctx);
//     }
    
//     HNSWIndex *index = HNSWIndex_New(1000, 128);

//     for (size_t i = 0; i < 1000; i++) {
//         for (size_t j = 0; j < 128; j++) {
//             float f[4] = {i, i, i, i};
//             HNSWIndex_AddVector(index, (const void *)f, i);
//         }
//     }
//     if (GetHNSWIndexSize(index) != 1000) {
//         return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
//     }
//     // The index is saved in the current directory from which redis is running.
//     char path [256];
//     SaveHNSWIndex(index, strcat(getcwd(path, sizeof(path)), "/index.ind"));
//     RemoveHNSWIndex(index);
//     HNSWIndex *loaded_index = HNSWIndex_New(2000, 128);
//     LoadHNSWIndex(loaded_index, path, 1000);
//     if (GetHNSWIndexSize(loaded_index) != 1000) {
//         return RedisModule_ReplyWithSimpleString(ctx, "Save/Load error");
//     }
//     return RedisModule_ReplyWithSimpleString(ctx, "OK");
// }

long long ustime(void) {
    struct timeval tv;
    long long ust;

    gettimeofday(&tv, NULL);
    ust = ((long long)tv.tv_sec) * 1000000;
    ust += tv.tv_usec;
    return ust;
}

int hnswlib_vector_search_million_test(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
    size_t n = 100000;
    int d = 128;
    VecSimAlgoParams params = {
        .hnswParams = {
            .initialCapacity = n,
            .M = 16,
            .efConstuction = 200
        },
        .algorithmType = VecSimAlgo_HNSW
    };
    VecSimIndex *index = VecSimIndex_New(&params, VecSimMetric_L2, VecSimVecType_FLOAT32, d);
    int k =11;

    RedisModule_Log(ctx, "warning", "creating vectors");
    float *vectors = RedisModule_Alloc(n*d*sizeof(float));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < d; j++) {
            (vectors+ i*d)[j] = i;
        }
    }
    RedisModule_Log(ctx, "warning", "adding vectors to index");
    for (size_t i = 0; i < n; i++) {
        VecSimIndex_AddVector(index, (const void *)(vectors+i*d), i);
    }
    if (VecSimIndex_IndexSize(index) != n) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    float query[128];
    for (size_t j = 0; j < d; j++) {
        query[j] = 50;
    }
    const long long start = ustime();
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index,  (const void *)query, k);
    const long long end = ustime();
    RedisModule_Log(ctx, "warning","Total time for %d-NN lookup : %llu microseconds\n", k, (end-start));
    RedisModule_Free(vectors);

    for (int i=0; i<11; i++) {
        int diff_id = ((int)(res[i].id - 50) > 0) ? (res[i].id - 50) : (50 - res[i].id);
        int dist = res[i].score;
        if ((diff_id != (i+1)/2) || (dist != (d*((i+1)/2)*((i+1)/2)))) {
            RedisModule_Log(ctx, "warning","Search test fail");
            return RedisModule_ReplyWithSimpleString(ctx, "Search test fail");

        }
    }
    return RedisModule_ReplyWithSimpleString(ctx, "OK");
}

int hnswlib_indexing_same_vector(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    if(argc != 1) {
        return RedisModule_WrongArity(ctx);
    }
        VecSimAlgoParams params = {
        .hnswParams = {
            .initialCapacity = 200,
            .M = 16,
            .efConstuction = 200
        },
        .algorithmType = VecSimAlgo_HNSW
    };
    VecSimIndex *index = VecSimIndex_New(&params, VecSimMetric_L2, VecSimVecType_FLOAT32, 4);

    for (size_t i = 0; i < 100; i++) {
        float num = i/10;
        float f[4] = {num, num, num, num};
        VecSimIndex_AddVector(index, (const void *)f, i);
    }
    if (VecSimIndex_IndexSize(index) != 100) {
        return RedisModule_ReplyWithSimpleString(ctx, "Vector add error");
    }
    // Run a query where all the results are supposed to be {5,5,5,5} (different ids).
    float query[4] = {4.9,4.95, 5.05, 5.1};
    VecSimQueryResult *res = VecSimIndex_TopKQuery(index,  (const void *)query, 10);
    for (int i=0; i<10; i++) {
        if (res[i].id < 50 || res[i].id >= 60 || res[i].score > 1) {
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
    if(
      RedisModule_CreateCommand(ctx, "vec_sim_test.hnswlib_search_million", hnswlib_vector_search_million_test,
        "",
        0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;

    if(
      RedisModule_CreateCommand(ctx, "vec_sim_test.hnswlib_indexing_same_vector", hnswlib_indexing_same_vector,
        "",
        0, 0, 0) == REDISMODULE_ERR)
        return REDISMODULE_ERR;
}
