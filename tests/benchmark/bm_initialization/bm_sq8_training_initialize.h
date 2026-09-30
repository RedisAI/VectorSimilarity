/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */
#pragma once

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSQ8Training, BM_Train, DATA_TYPE_INDEX_T)
(benchmark::State &st) { Train(st); }
BENCHMARK_REGISTER_F(BM_VecSimSQ8Training, BM_Train)
    ->Args({1024})
    ->Args({5000})
    ->ArgNames({"training_threshold"})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(2);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSQ8Training, BM_TrainAsync, DATA_TYPE_INDEX_T)
(benchmark::State &st) { TrainAsync(st); }
BENCHMARK_REGISTER_F(BM_VecSimSQ8Training, BM_TrainAsync)
    ->ArgsProduct({{1024, 5000}, {4, 8}})
    ->ArgNames({"training_threshold", "thread_count"})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(2);

BENCHMARK_TEMPLATE_DEFINE_F(BM_VecSimSQ8Training, BM_AddVectorsDuringInitialIngest,
                            DATA_TYPE_INDEX_T)
(benchmark::State &st) { AddVectorsDuringInitialIngest(st); }
BENCHMARK_REGISTER_F(BM_VecSimSQ8Training, BM_AddVectorsDuringInitialIngest)
    ->ArgsProduct({{1024, 5000}, {4, 8}})
    ->ArgNames({"training_threshold", "thread_count"})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->Iterations(1);
